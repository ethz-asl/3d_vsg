import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from python_tsp.heuristics import solve_tsp_simulated_annealing

from config import DatasetCfg
from src.dataset import SceneGraphChangeDataset
from src.models import SimpleMPGNN, FocalLoss
from src.utils.extract_data import build_scene_graph, transform_locations


def calc_path_cost(path, node_data_1, node_data_2, var_label, n_obj):
    cost = 0
    changes = 0
    node_2_dict = {node[0]:node[1] for node in node_data_2}
    id_paths = [node_data_1[idx][0] for idx in path]
    for j in range(len(id_paths) - 1):
        # Calculates distance between two object waypoints in trajectory
        dist1 = node_2_dict[id_paths[j]]["attributes"]["location"] if id_paths[j] in node_2_dict.keys() else \
                node_data_1[j][1]["attributes"]["location"]
        dist2 = node_2_dict[id_paths[j+1]]["attributes"]["location"] if id_paths[j+1] in node_2_dict.keys() else \
            node_data_1[j+1][1]["attributes"]["location"]

        # Adds to total path length
        cost += torch.norm(dist1-dist2)
        changes += var_label[j+1]

        # If ith object is found, return path length
        if changes >= n_obj:
            return cost, changes
    return cost, changes


def eval_change_detection(dataset, model, scan_root):
    seg_root = os.path.join(scan_root, "semantic_segmentation_data")
    diff_arr = np.zeros((5, 2, 3))
    better = 0
    all = 0
    for data in tqdm(dataset):
        if data.num_nodes > 2:
            # Process data for scene
            pred_i = model(data.x, data.edge_index.type(torch.LongTensor), data.edge_attr)

            _, nodes_1, edges_1 = build_scene_graph(dataset.objects_dict, dataset.relationships_dict,
                                                    data.input_graph, seg_root)
            _, nodes_2, edges_2 = build_scene_graph(dataset.objects_dict, dataset.relationships_dict,
                                                    data.output_graph, seg_root)
            transf_node_1 = transform_locations(nodes_1, data.input_tf)
            transf_node_2 = transform_locations(nodes_2, data.output_tf)

            # build distance matrix of objects
            dist_mat = np.zeros((data.num_nodes, data.num_nodes))
            for i in range(data.num_nodes):
                obj1_pos = transf_node_1[i][1]["attributes"]["location"]
                for j in range(i+1, data.num_nodes):
                    obj2_pos = transf_node_1[j][1]["attributes"]["location"]
                    dist = torch.norm(obj1_pos[:2, 0]-obj2_pos[:2, 0])
                    dist_mat[i, j] = dist
                    dist_mat[j, i] = dist

            # Get list of objects with change
            change_arr = torch.sum(data.y[:, :2], dim=1)
            changed_obj = torch.sum(change_arr > 0)
            if diff_arr.shape[0] < changed_obj:
                diff_arr = np.concatenate((diff_arr, np.zeros((changed_obj-diff_arr.shape[0], 2, 3))))

            # Naive Path Planning
            naive_path, _ = solve_tsp_simulated_annealing(dist_mat)

            # Uncertainty Aware Path Planning
            probs = torch.sigmoid(pred_i)
            sums = torch.sum(probs, dim=1)
            sorted, indices = torch.sort(sums, descending=True)

            # Evaluation
            for i in range(changed_obj-1):
                # Full Coverage: calculates travel distance along full coverage trajectory to ith nearest changed object
                naive_dist, outcome = calc_path_cost(naive_path, transf_node_1, transf_node_2, change_arr, i)
                diff_arr[i, 0, 0] += naive_dist
                diff_arr[i, 0, 1] += 1
                diff_arr[i, 0, 2] += naive_dist**2

                # DeltaVSG: Visits a subset of objects with highest change probability calculated by DeltaVSG
                n = min(i + 4, indices.shape[0])
                to_visit = [indices[i].item() for i in range(n)]
                to_visit.append(0)
                to_visit_full = list(set(to_visit))
                aware_dist_mat = dist_mat[to_visit_full, :][:, to_visit_full]
                aware_path, _ = solve_tsp_simulated_annealing(aware_dist_mat)
                aware_path = [to_visit_full[idx] for idx in aware_path]

                # Calculates travel distance along this trajectory to ith nearest object
                aware_dist, n_changes = calc_path_cost(aware_path, transf_node_1, transf_node_2, change_arr[aware_path], i)

                # If the requisite # of objects not found, visits all objects (following full coverage trajectory)
                # to nearest changed object
                if n_changes < i:
                    new_to_visit = [idx for idx in range(data.num_nodes) if idx not in to_visit_full]
                    last_node = aware_path[-1]
                    new_to_visit.append(last_node)
                    new_to_visit_full = list(set(new_to_visit))
                    new_naive_mat = dist_mat[new_to_visit_full, :][:, new_to_visit_full]
                    new_naive_path, _ = solve_tsp_simulated_annealing(new_naive_mat)
                    new_naive_path = [new_to_visit_full[idx] for idx in new_naive_path]
                    start_idx = new_naive_path.index(last_node)
                    path_from_last = new_naive_path[start_idx:] + new_naive_path[:start_idx]
                    new_naive_dist, _ = calc_path_cost(path_from_last, transf_node_1, transf_node_2, change_arr[path_from_last], i-n_changes)
                    aware_dist += new_naive_dist

                diff_arr[i, 1, 0] += aware_dist
                diff_arr[i, 1, 1] += 1
                diff_arr[i, 1, 2] += aware_dist ** 2
                if aware_dist < naive_dist:
                    better += 1
                all += 1

    print("{:.3f}% of better runs".format(100*better/all))
    print("Average Coverage Dist: {:.3f}".format(np.sum(diff_arr[:, 0, 0])/all))
    print("Average DeltaVSG Dist: {:.3f}".format(np.sum(diff_arr[:, 1, 0])/all))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Inference on {}".format(device))

    # Load Demo Data
    dataset_cfg = DatasetCfg()
    dataset = SceneGraphChangeDataset(cfg=dataset_cfg)
    dataset._load_raw_files()

    with open(dataset_cfg.task_sim.split_path, 'rb') as handle:
        splits = pickle.load(handle)
    test_idx = [i for i in range(len(dataset)) if dataset[i]["scene"] in splits["test"]]
    hyperparams = {
        "model_name": "DeltaVSG",
        "model_type": "gnn",
        "hidden_layers": [32],
        "lr": 0.001,
        "weight_decay": 5e-4,
        "num_epochs": 10,
        "bs": 16,
        "loss": FocalLoss(0.8),
        "results_path": dataset_cfg.results_path
    }

    model = SimpleMPGNN(dataset.num_node_features, dataset.num_classes, dataset.num_edge_features,
                        hyperparams["hidden_layers"])
    model.load_state_dict(torch.load(dataset_cfg.task_sim.model_path, map_location=device))
    model.eval()

    eval_change_detection(dataset[test_idx], model, os.path.join(dataset_cfg.root, "raw"))

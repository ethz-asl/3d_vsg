import os
import torch
from tqdm import tqdm
from config import DatasetCfg
from src.dataset import SceneGraphChangeDataset
from src.models import SimpleMPGNN, SimpleMLP, SimpleSA, PPNBaseline
from src.utils import build_scene_graph, visualize_graph, visualize_results, get_dataset_files


def vis_test_sample(scan_dir, in_scan_id, out_scan_id, predicted_var, actual_var, vis_io=False):
    ply_file = "labels.instances.annotated.v2.ply"
    objects_dict, relationships_dict = get_dataset_files(scan_dir)
    sem_seg_dir = os.path.join(scan_dir, "semantic_segmentation_data")
    # Visualize Graph
    in_mesh_path = os.path.join(sem_seg_dir, in_scan_id, ply_file)
    in_graph, in_nodes, in_edges = build_scene_graph(objects_dict, relationships_dict, in_scan_id, sem_seg_dir)
    if vis_io:
        print("Visualizing Input Graph")
        visualize_graph(in_mesh_path, in_nodes, in_edges)

    if vis_io:
        print("Visualizing Output Graph")
        out_mesh_path = os.path.join(sem_seg_dir, out_scan_id, ply_file)
        out_graph, out_nodes, out_edges = build_scene_graph(objects_dict, relationships_dict, out_scan_id, sem_seg_dir)
        visualize_graph(out_mesh_path, out_nodes, out_edges)

    # Visualizing Results
    result_type = "Instance"
    print("Visualizing {} Variability Results".format(result_type))
    visualize_results(in_mesh_path, in_nodes, predicted_var, actual_var, result_type)


def inference_vis(dataset, model, nnet_type, scan_dir):
    start_idx = 39
    for i in tqdm(range(start_idx, len(dataset))):
        data = dataset[i]

        if nnet_type == "gnn":
            e_idx_i = data.edge_index.type(torch.LongTensor)
            e_att_i = data.edge_attr
            pred_i = model(data.x, e_idx_i, e_att_i)
        else:
            pred_i = model(data.x)

        # Get Instance Variability from current embedding (1=object in scene, 0=object out of scene)
        label = data.y
        vis_test_sample(scan_dir, data.input_graph, data.output_graph, pred_i, label)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Inference on {}".format(device))

    dataset_cfg = DatasetCfg()
    dataset = SceneGraphChangeDataset(cfg=dataset_cfg)

    splits = []
    for i in range(len(dataset)):
        in_graph = dataset[i].input_graph
        out_graph = dataset[i].output_graph
        if in_graph in dataset_cfg.inference.scans and out_graph in dataset_cfg.inference.scans:
            splits.append(i)

    demo_set = dataset[splits]
    model_path = dataset_cfg.inference.model_path

    hyperparams = {
        "model_name": "DeltaVSG Baseline",
        "model_type": "gnn",
        "hidden_layers": [32, 32],
        "results_path": dataset_cfg.results_path
    }

    model = SimpleMPGNN(dataset.num_node_features, dataset.num_classes, dataset.num_edge_features,
                        hyperparams["hidden_layers"])
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    # Download demo scene meshes
    inference_root = os.path.join(dataset_cfg.root, "raw")
    script_path = os.path.join(dataset_cfg.root, "download_3rscan.py")
    raw_data_path = os.path.join(inference_root, "semantic_segmentation_data")
    meshfile = "labels.instances.annotated.v2.ply"
    for scan in dataset_cfg.inference.scans:
        if not os.path.isfile(os.path.join(raw_data_path, scan, meshfile)):
            os.system("python {} --out_dir {} --type {} --id {}".format(script_path, raw_data_path, meshfile, scan))

    inference_vis(demo_set, model, hyperparams["model_type"], inference_root)



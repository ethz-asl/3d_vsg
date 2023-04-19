import torch
import pickle
from tqdm import tqdm

from config import DatasetCfg
from src.utils import EvalLogger
from src.dataset import SceneGraphChangeDataset
from src.models import SimpleMPGNN, SimpleMLP, SimpleSA, PPNBaseline, FocalLoss


def calculate_conf_mat(pred, label):
    class_preds = (pred > 0.).float()
    tp = torch.sum(torch.minimum(class_preds, label))
    tn = label.shape[0] - torch.sum(torch.maximum(class_preds, label))
    fp = torch.sum(class_preds) - tp
    fn = torch.sum(label) - tp
    return torch.tensor([[tp, fp], [fn, tn]])


def eval_var(dataset, model, nnet_type, title, dev):
    logger = EvalLogger(title)
    for data in tqdm(dataset):
        if nnet_type == "gnn":
            e_idx_i = data.edge_index.type(torch.LongTensor)
            e_att_i = data.edge_attr
            pred_i = model(data.x.to(dev), e_idx_i.to(dev), e_att_i.to(dev))
        else:
            pred_i = model(data.x.to(dev))

        state_valid = pred_i[torch.nonzero(data.state_mask == 1), 0]
        state_valid_label = data.y[torch.nonzero(data.state_mask == 1), 0]

        pos_valid = pred_i[torch.nonzero(data.y[:, 2] == 0), 1]
        pos_valid_label = data.y[torch.nonzero(data.y[:, 2] == 0), 1]

        node_valid = pred_i[:, 2]
        node_valid_label = data.y[:, 2]

        state_conf = calculate_conf_mat(state_valid.to(dev), state_valid_label.to(dev))
        pos_conf = calculate_conf_mat(pos_valid.to(dev), pos_valid_label.to(dev))
        node_conf = calculate_conf_mat(node_valid.to(dev), node_valid_label.to(dev))

        logger.log_eval_iter(state_conf, pos_conf, node_conf)

    logger.print_eval_results()


if __name__ == "__main__":
    dataset_cfg = DatasetCfg()
    dataset_cfg.splits_path = "./results/baseline_splits.pickle"
    dataset_cfg.root = "./data/training/"
    model_path = "./pretrained/deltavsg_baseline.pt"
    
    dataset = SceneGraphChangeDataset(cfg=dataset_cfg)

    with open(dataset_cfg.splits_path, 'rb') as handle:
        splits = pickle.load(handle)
    test_idx = [i for i in range(len(dataset)) if dataset[i]["scene"] in splits["test"]]

    hyperparams = {
        "model_name": "DeltaVSG Baseline",
        "model_type": "gnn",
        "hidden_layers": [32, 32],
        "results_path": dataset_cfg.results_path
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define DeltaVSG model
    model = SimpleMPGNN(dataset.num_node_features, dataset.num_classes, dataset.num_edge_features,
                        hyperparams["hidden_layers"]).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    eval_var(dataset[test_idx], model, hyperparams["model_type"], hyperparams["model_name"], device)

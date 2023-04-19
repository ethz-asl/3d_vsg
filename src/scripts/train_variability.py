import torch
import random
import pickle
from tqdm import tqdm
from torch_geometric.data import DataLoader

from config import DatasetCfg
from src.dataset import SceneGraphChangeDataset
from src.utils import TrainingLogger, calculate_conf_mat
from src.models import SimpleMPGNN, SimpleMLP, SimpleSA, PPNBaseline, FocalLoss


def calculate_training_loss(data, nnet, l_fn, nnet_type, loss_dev):
    # Performs forward pass, and calculates training loss for a given batch
    x_i = data.x.to(loss_dev)
    node_mask = 1 - data.y[:, 2].to(loss_dev)
    state_mask = data.state_mask.to(loss_dev)

    # Performs forward pass for network
    if nnet_type == "gnn":
        e_idx_i = data.edge_index.type(torch.LongTensor).to(loss_dev)
        e_att_i = data.edge_attr.to(loss_dev)
        pred_i = nnet(x_i, e_idx_i, e_att_i)
    else:
        pred_i = nnet(x_i)

    # Computes loss for batch given masks for invalid samples
    loss_tensor = l_fn(pred_i, data.y.to(loss_dev).to(torch.float32))
    loss_state = loss_tensor[:, 0]
    loss_pos = loss_tensor[:, 1]
    loss_mask = loss_tensor[:, 2]
    loss = (torch.sum(torch.multiply(loss_state, state_mask)) + torch.sum(torch.multiply(loss_pos, node_mask)) +
            torch.sum(loss_mask)) / (torch.sum(node_mask) + torch.sum(state_mask) + torch.numel(node_mask))
    return loss, pred_i


def train_neuralnet(dset, neuralnet, hyperparams, train_dev):
    # Define Training and Validation split
    train_n = int(len(dset) * 0.85)
    train_set, val_set = torch.utils.data.random_split(dset,  [train_n, len(dset)-train_n])
    train_loader = DataLoader(train_set, batch_size=hyperparams["bs"])
    val_loader = DataLoader(val_set, batch_size=1)
    # Define loss and training params
    l_fn = hyperparams["loss"]
    epochs = hyperparams["num_epochs"]
    nnet_type = hyperparams["model_type"]
    optimizer = torch.optim.Adam(neuralnet.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
    # Initialize Logger
    logger = TrainingLogger(hyperparams["model_name"], int(train_n/hyperparams["bs"]), 20, hyperparams["results_path"], True)
    neuralnet.train()

    # Main Training Loop
    for i in range(epochs):
        print("Training. Epoch {}".format(i+1))
        for data in train_loader:
            optimizer.zero_grad()
            # Performs forward pass and calculates training loss
            loss, _ = calculate_training_loss(data, neuralnet, l_fn, nnet_type, train_dev)

            # Logs training iteration
            logger.log_training_iter(loss.item())

            # Performs backward pass
            loss.backward()
            optimizer.step()

        # Logs training epoch
        logger.log_training_epoch()

        print("Validation:")
        neuralnet.eval()
        for data in tqdm(val_loader):
            # Performs forward pass and calculates validation loss for given batch
            loss, pred_i = calculate_training_loss(data, neuralnet, l_fn, nnet_type, train_dev)
            val_loss = loss.item()

            # Calculates accuracy metrics for given batch
            state_valid = pred_i[torch.nonzero(data.state_mask == 1), 0]
            state_valid_label = data.y[torch.nonzero(data.state_mask == 1), 0].to(train_dev)
            state_conf = calculate_conf_mat(state_valid, state_valid_label)

            pos_valid = pred_i[torch.nonzero(data.y[:, 2] == 0), 1]
            pos_valid_label = data.y[torch.nonzero(data.y[:, 2] == 0), 1].to(train_dev)
            pos_conf = calculate_conf_mat(pos_valid, pos_valid_label)

            node_valid = pred_i[:, 2]
            node_valid_label = data.y[:, 2].to(train_dev)
            node_conf = calculate_conf_mat(node_valid, node_valid_label)

            # Logs validation iteration
            logger.log_validation_iter(val_loss, state_conf, pos_conf, node_conf)

        # Logs validation epoch
        logger.print_val_results()
        logger.log_validation_epoch(neuralnet)

    # Final training results and plots
    logger.plot_training_losses()
    logger.plot_valid_losses()
    logger.plot_accuracies()


if __name__ == "__main__":
    # Load Dataset
    dataset_cfg = DatasetCfg()
    dataset = SceneGraphChangeDataset(cfg=dataset_cfg)

    # Calculate Class Distribution on training set & set positive weights for class imbalance
    pos = torch.sum(dataset.data.y.to(torch.float32), dim=0)
    val_state = torch.sum(dataset.data.state_mask)
    val_pos = dataset.data.y.shape[0] - pos[2]
    val_node = dataset.data.y.shape[0]
    ratios = torch.tensor([pos[0]/val_state, pos[1]/val_pos, pos[2]/val_node])
    weights = torch.divide(torch.ones((3,)), ratios)
    print("Class distribution: {}".format(ratios))
    
    #  Generate Train, Validation, and Test splits
    load = False
    if load:
        with open(dataset_cfg.splits_path, 'rb') as handle:
            splits = pickle.load(handle)
    else:
        scenes = list(set([sample["scene"] for sample in dataset]))
        test_split = random.sample(range(len(scenes)), int(len(scenes)*0.15))
        splits = {"tv": [scenes[i] for i in range(len(scenes)) if i not in test_split],
                  "test": [scenes[i] for i in range(len(scenes)) if i in test_split]}
        with open(dataset_cfg.splits_path, 'wb') as handle:
            pickle.dump(splits, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    tv_idx = [i for i in range(len(dataset)) if dataset[i]["scene"] in splits["tv"]]
    tv_set = dataset[tv_idx]

    # Move training to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    print("Training on {}".format(device))

    # Define training hyperparameters
    hyperparams = {
        "model_name": "DeltaVSG baseline",
        "model_type": "gnn",
        "hidden_layers": [32],
        "lr": 0.001,
        "weight_decay": 0,
        "num_epochs": 20,
        "bs": 16,
        "loss": FocalLoss(1.1, weights.to(device)),
        "results_path": dataset_cfg.results_path
    }

    # Define DeltaVSG model
    model = SimpleMPGNN(tv_set.num_node_features, tv_set.num_classes, tv_set.num_edge_features,
                        hyperparams["hidden_layers"]).to(device)
    # Train model
    train_neuralnet(tv_set, model, hyperparams, device)

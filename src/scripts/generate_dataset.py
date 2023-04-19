import os
import torch

from config import DatasetCfg
from src.dataset import SceneGraphChangeDataset


def check_data_reqs(data_root):
    data_req_filename = os.path.join(data_root, "raw", "raw_files.txt")
    if not os.path.isfile(data_req_filename):
        # Generate data requirements file if file does not exist
        required_files = ["3RScan.json", "semantic_segmentation_data", "3DSSG/affordances.txt",
                          "3DSSG/attributes.txt", "3DSSG/classes.txt", "3DSSG/objects.json",
                          "3DSSG/relationships.txt", "3DSSG/relationships.json"]

        with open(data_req_filename, 'w') as f:
            for filename in required_files:
                f.write("{}\n".format(filename))


if __name__ == "__main__":
    # Load Dataset
    dataset_cfg = DatasetCfg()
    dataset_cfg.load = False
    check_data_reqs(dataset_cfg.root)

    dataset = SceneGraphChangeDataset(cfg=dataset_cfg)

    print("Generated new dataset")
    print("File location: {}".format(os.path.join(dataset.root, "processed")))
    print("PCA Enabled: {}".format(dataset_cfg.pca_enable))
    print("# Features: {}".format(dataset.data.x.shape[1]))

    pos_embed = "None"
    if dataset_cfg.attributes.global_loc:
        pos_embed = "Global"
    if dataset_cfg.relationships.relative_loc:
        pos_embed = "Relative"
    print("Position Embedding: {}".format(pos_embed))
    print("Object Taxonomy: {}".format(dataset_cfg.objects.taxonomy if dataset_cfg.objects.enabled else "None"))
    print("Variability Threshold: {}".format(dataset_cfg.variability.threshold))

    # Calculate Class Distribution on training set & set positive weights for class imbalance
    pos = torch.sum(dataset.data.y.to(torch.float32), dim=0)
    val_state = torch.sum(dataset.data.state_mask)
    val_pos = dataset.data.y.shape[0] - pos[2]
    val_node = dataset.data.y.shape[0]
    ratios = torch.tensor([pos[0]/val_state, pos[1]/val_pos, pos[2]/val_node])

    print("Class distribution: {}".format(ratios))

import os
import json
import time
import torch
from typing import List, Dict
from torch_geometric.data import InMemoryDataset, Data

from config import DatasetCfg
from src.utils import build_scene_graph, get_change_list, get_scene_list, get_dataset_files
from src.dataset import BinaryNodeEmbedding, BinaryVariabilityEmbedding, BinaryEdgeEmbedding, PCADimReduction


class SceneGraphChangeDataset(InMemoryDataset):
    """ 3D Scene Graph Variability Estimation Dataset (PyTorch Geometric dataset)
    Reads raw data from 3DSSG dataset (Wald et al.) and formulates a supervised graph-based learning dataset """
    def __init__(self, root=None, cfg: DatasetCfg = None, transform=None, pre_transform=None, pre_filter=None):
        self.cfg: DatasetCfg = cfg
        self.root = root if root else self.cfg.root
        self.seg_data_root = os.path.join(self.root, "raw", "semantic_segmentation_data")
        self.raw_files: str = os.path.join(self.root, "raw", "raw_files.txt")

        if not cfg.load:
            # If load set to False, will move any processed dataset to a new repo
            if os.path.isdir(os.path.join(self.root, "processed")):
                if len(os.listdir(os.path.join(self.root, "processed"))) > 0:
                    old_root = os.path.join(self.root, "old_processed")
                    if not os.path.exists(old_root):
                        os.makedirs(old_root)
                    new_folder_name = time.strftime("%Y_%m_%d_%H_%M_%S")
                    os.rename(os.path.join(self.root, "processed"), os.path.join(old_root, new_folder_name))

        super().__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _load_raw_files(self):
        # Load raw data files as per standard dataset folder organization
        if self.cfg:
            root: str = self.cfg.root
        else:
            root = self.root

        scan_dir = os.path.join(root, "raw")
        self.scans: List[Dict] = json.load(open(os.path.join(scan_dir, "3RScan.json")))
        self.objects_dict, self.relationships_dict = get_dataset_files(scan_dir)

        self.node_embedder: BinaryNodeEmbedding = BinaryNodeEmbedding(att_cfg=DatasetCfg.attributes, obj_cfg=DatasetCfg.objects)
        self.edge_embedder: BinaryEdgeEmbedding = BinaryEdgeEmbedding(cfg=DatasetCfg.relationships)
        self.variability_embedder: BinaryVariabilityEmbedding = BinaryVariabilityEmbedding(cfg=DatasetCfg.variability)

    @property
    def raw_file_names(self):
        if os.path.isfile(self.raw_files):
            with open(self.raw_files) as f:
                files = f.read().splitlines()
        else:
            raise Exception("Raw File List Not Found")
        return files

    @property
    def processed_file_names(self):
        return ['scene_graph_data.pt']

    def download(self):
        raise Exception("Files Not Found. Download dataset files as per standard format")

    def process(self):
        self._load_raw_files()
        samples = []
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x: (_ for _ in x)

        scene_count_arr = []
        num_scenes = []
        bad_scenes = 0
        # Iterate through all scenes in dataset
        for scene in tqdm(self.scans):
            scan_id_set, scan_tf_set, scan_changes_set = get_scene_list(scene)
            scene_count = 0
            # Iterate through all scans for given scene (note: augments data by assuming symmetry and time indep.)
            for i in range(len(scan_id_set)):
                for j in range(len(scan_id_set)):
                    if i != j:
                        scene_count += 1
                        # Extract nodes and edges from scene graph data for input and output
                        _, nodes_1, edges_1 = build_scene_graph(self.objects_dict, self.relationships_dict,
                                                                scan_id_set[i], self.seg_data_root)
                        _, nodes_2, edges_2 = build_scene_graph(self.objects_dict, self.relationships_dict,
                                                                scan_id_set[j], self.seg_data_root)

                        if nodes_1 is not None and nodes_2 is not None:
                            # Transform objects into common reference frame
                            T_1I = scan_tf_set[i]
                            T_2I = scan_tf_set[j]

                            # Apply Node (Input) Embedding
                            node_embeddings, node_idxs, node_pos, node_classifications = \
                                self.node_embedder.generate_node_embeddings(nodes_1)

                            # Apply Edge (Input) Embedding
                            edge_embeddings, edge_idxs = self.edge_embedder.generate_edge_embeddings(nodes_1, edges_1,
                                                                                                     node_idxs)

                            # Apply Variability (Label) Embedding
                            change_list = get_change_list(scan_changes_set[i], scan_changes_set[j])
                            node_labels, state_mask = self.variability_embedder.generate_variability_embedding(
                                nodes_1, nodes_2, node_idxs, change_list)

                            # Formulate Pytorch Geometry data sample
                            sample = Data(
                                x=node_embeddings,
                                edge_index=edge_idxs,
                                edge_attr=edge_embeddings,
                                y=node_labels,
                                pos=node_pos,
                                classifications=node_classifications,
                                input_graph=scan_id_set[i],
                                output_graph=scan_id_set[j],
                                input_tf=T_1I,
                                output_tf=T_2I,
                                state_mask=state_mask,
                                scene=scene["reference"],
                                fw_mask=(i < j)
                            )
                            samples.append(sample)
                        else:
                            bad_scenes += 1
            scene_count_arr.append(scene_count)
            num_scenes.append(len(scan_id_set))

        data, slices = self.collate(samples)

        # Apply PCA dimensionality reduction
        if self.cfg.pca_enable:
            pca_transform = PCADimReduction()
            data.x = pca_transform(data.x, self.cfg.pca_num_features)

        # Save dataset
        torch.save((data, slices), self.processed_paths[0])


import torch
import itertools
from typing import List, Tuple

from config import DatasetCfg
from src.dataset.Relationships3DSSG import Relationships3DSSG


def get_relative_dist(node_1: Tuple, node_2: Tuple) -> torch.Tensor:
    # Calculates relative distance between two objects in a scene graph
    return torch.Tensor(node_1[1]["attributes"]["location"] - node_2[1]["attributes"]["location"])


class BinaryEdgeEmbedding:
    """ Embeds edge relationships using binary encoding. Can be configured with cfg file"""
    def __init__(self, cfg: DatasetCfg.RelationshipParams):
        self.cfg: DatasetCfg.RelationshipParams = cfg
        self.loc: bool = cfg.relative_loc
        self.n: int = len(Relationships3DSSG)

    def generate_edge_embeddings(self, nodes_1: List[Tuple],
                                 edges: List[List],
                                 node_idx: List[int]) -> (torch.Tensor, torch.Tensor):
        # Generates matrix of edge embedding matrices (relationship encoding + correspondence matrices) for a given scan

        node_local_id_to_idx = {int(node[0]): idx for (idx, node) in enumerate(nodes_1)}
        num_nodes = len(node_idx)
        num_edges = num_nodes * (num_nodes - 1)
        relative_loc = torch.zeros(num_edges, 3, dtype=torch.float32)
        _edge_idx_tensor = torch.zeros((2, num_edges), dtype=torch.int64)
        _edge_embeds_tensor = torch.zeros(num_edges, self.n, dtype=torch.int64)
        edge_map = {}

        # Creates edges for every combination of objects in the scene with their relative positions
        for (idx, (fr, to)) in enumerate(itertools.permutations(range(num_nodes), 2)):
            # (0,1), ...(0,9), (1,0), (1,2), ... (1,9), ...
            _edge_idx_tensor[0, idx] = fr
            _edge_idx_tensor[1, idx] = to
            relative_loc[idx, :] = get_relative_dist(nodes_1[fr], nodes_1[to]).flatten()
            edge_map[(fr, to)] = idx

        # Creates edges for every semantic relationship in the scene graph
        for edge in edges:
            e1 = node_local_id_to_idx[edge[0]]
            e2 = node_local_id_to_idx[edge[1]]
            if e1 != e2:
                # Binary encoding for semantic relationship
                rel = edge[2]-1
                idx = edge_map[(e1, e2)]
                _edge_embeds_tensor[idx, rel] = 1

        # Keeps edges with semantic relationships, and edges with geometric relationship within some threshold,
        # depending on config file
        if self.loc:
            # Normalize using the largest distance found in graph
            dist = torch.norm(relative_loc, p=2, dim=1)
            thresh = torch.quantile(dist, self.cfg.loc_threshold)
            valid_edges = torch.logical_or(_edge_embeds_tensor.sum(dim=1) > 0, thresh > dist)
            _edge_embeds_tensor = torch.hstack([_edge_embeds_tensor[valid_edges, :], relative_loc[valid_edges, :]])
            _edge_idx_tensor = _edge_idx_tensor[:, valid_edges]
        else:
            _edge_embeds_tensor = _edge_embeds_tensor.type(torch.FloatTensor)
        return _edge_embeds_tensor, _edge_idx_tensor

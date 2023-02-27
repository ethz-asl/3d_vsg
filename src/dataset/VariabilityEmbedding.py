from config.DatasetCfg import DatasetCfg
import logging
import torch
from typing import List, Dict, Tuple
logger = logging.getLogger(__name__)


def get_relative_dist(node_1: Tuple, node_2: Tuple) -> torch.Tensor:
    # Calculates relative distance between two objects in a scene graph
    return torch.Tensor(node_1[1]["attributes"]["location"] - node_2[1]["attributes"]["location"])


class BinaryVariabilityEmbedding:
    """ Embeds variability labels from a pair of scans from the same scene. Can be configured with cfg file"""
    def __init__(self, cfg: DatasetCfg.VariabilityParams):
        self.cfg: DatasetCfg.VariabilityParams = cfg
        self.threshold: float = self.cfg.threshold

    def generate_variability_embedding(self, nodes_1: List[Tuple], nodes_2: List[Tuple], input_node_idx: List[int], change_set: List) -> Tuple:
        # Formats node lists and establishes correspondences
        output_node_idx = [val[0] for val in nodes_2]
        output_nodes = [val for val in nodes_2]
        input_nodes = [val for val in nodes_1]
        output_embeddings = []
        state_mask = []

        # Iterates through all input nodes & calculates variability label
        for i in range(len(input_node_idx)):
            idx = input_node_idx[i]
            if idx in output_node_idx:
                # If object correspondence exists, calculate node & position variability
                output_node = output_nodes[output_node_idx.index(idx)]
                input_node = input_nodes[i]
                state_var, state_available = self.state_var(input_node, output_node)
                pos_var = (idx in change_set)
                var_embed = torch.tensor([[state_var, pos_var, 0]])
                output_embeddings.append(var_embed)
                # Also generates mask determining whether a state was labeled for this sample
                state_mask.append(state_available)
            else:
                # If no object correspondence exists, set label for instance variability
                output_embeddings.append(torch.tensor([[0, 0, 1]]))
                state_mask.append(0)

        output_embeddings_tensor = torch.cat(output_embeddings)
        state_mask_tensor = torch.tensor(state_mask)
        return output_embeddings_tensor, state_mask_tensor

    def state_var(self, in_node: Tuple, out_node: Tuple) -> Tuple:
        # Calculates state and position variability for nodes with correspondence

        # Sets state variability to 1 if state changes from initial to future scene
        state_diff = 0
        if "state" in in_node[1]["attributes"].keys() and "state" in out_node[1]["attributes"].keys():
            state_diff = int(in_node[1]["attributes"]["state"] != out_node[1]["attributes"]["state"])
            state_available = 1
        else:
            state_available = 0

        return state_diff, state_available

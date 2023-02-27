from src.dataset.Attributes3DSSG import Attributes3DSSG
from src.dataset.ObjectClassification import object_classifications
import logging
from config import DatasetCfg
import torch
from typing import List, Dict, Tuple
logger = logging.getLogger(__name__)


class BinaryNodeEmbedding:
    """ Embeds node attributes using binary encoding. Can be configured with att_cfg and obj_cfg files"""
    def __init__(self, att_cfg: DatasetCfg.AttributeParams, obj_cfg: DatasetCfg.ObjectParams):
        self.att_cfg: DatasetCfg.AttributeParams = att_cfg
        self.obj_cfg: DatasetCfg.ObjectParams = obj_cfg
        self.taxonomy = obj_cfg.taxonomy
        self.n_obj = obj_cfg.n_obj
        self.obj_enabled = obj_cfg.enabled
        self.loc: bool = att_cfg.global_loc

    def generate_node_embeddings(self, node_list: List[Tuple]) -> (torch.Tensor, List):
        # Generates tensor of embeddings for a single graph sample

        # Formulates dictionary of node attributes and extracts unique node IDs for correspondence
        node_dict = {node[0]: node[1] for node in node_list}
        node_ids = sorted(list(node_dict.keys()))
        num_nodes = len(node_ids)

        # Initializes embedding vectors
        attribute_embeddings = torch.zeros((num_nodes, len(Attributes3DSSG)), dtype=torch.float)
        object_embeddings = torch.zeros((num_nodes, self.n_obj), dtype=torch.float)
        node_locations = torch.zeros((num_nodes, 3), dtype=torch.float)
        node_classifications = torch.zeros((num_nodes, 1), dtype=torch.float)

        # Iterates through nodes in scan
        for idx, node in enumerate(node_ids):
            # Generates binary embedding for semantic attributes
            attribute_embeddings[idx, :] = self.calc_attribute_embedding(node_dict[node]["attributes"])
            # Generates binary embedding for object classes
            if self.obj_enabled:
                object_embeddings[idx, :] = self.calc_object_embedding(node_dict[node]["label"])
            # Extracts position vector
            node_locations[idx, :] = node_dict[node]["attributes"]["location"].flatten()
            # Extract global class ID
            node_classifications[idx] = int(node_dict[node]["global_id"])

        # Add location and object class to attribute vector depending on configuration
        if self.loc:
            attribute_embeddings = torch.hstack((attribute_embeddings, node_locations))
        if self.obj_enabled:
            attribute_embeddings = torch.hstack((attribute_embeddings, object_embeddings))

        return attribute_embeddings, node_ids, node_locations, node_classifications

    def calc_attribute_embedding(self, node_dict: Dict) -> torch.Tensor:
        # Embed semantic attributes given taxonomy (Attributes 3DSSG)

        # Get list of semantic attributes
        raw_embedding: List[str] = []
        for (key, val) in node_dict.items():
            if key != "location":
                for v in val:
                    raw_embedding.append("_".join([key, v]))

        # Format raw attributes and format binary vector according to taxonomy
        raw_embedding = [str(Attributes3DSSG.to_enum(r)) for r in raw_embedding]
        torch_embedding = torch.tensor(Attributes3DSSG.binary_encode(raw_embedding))
        return torch.unsqueeze(torch_embedding, 0)

    def calc_object_embedding(self, label):
        # Embed object attributes given taxonomy
        embedding = torch.zeros((1, self.n_obj))
        object_class = [obj_class for obj_class in object_classifications if obj_class.label == label][0]
        if self.taxonomy == "rio":
            embedding[0, object_class.rio] = 1
        elif self.taxonomy == "rio27":
            embedding[0, object_class.rio27] = 1
        elif self.taxonomy == "eigen13":
            embedding[0, object_class.eigen13] = 1
        elif self.taxonomy == "nyu40":
            embedding[0, object_class.nyu40] = 1
        return embedding

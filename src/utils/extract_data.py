import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import networkx as nx


def get_change_list(changes1, changes2):
    more_changes, fewer_changes = (changes1.copy(), changes2.copy()) if len(changes1) > len (changes2) else (changes2.copy(), changes1.copy())
    return list(set(more_changes[len(fewer_changes):]))


def get_dataset_files(scan_dir):
    object_data = json.load(open(os.path.join(scan_dir, "3DSSG", "objects.json")))
    relationship_data = json.load(open(os.path.join(scan_dir, "3DSSG", "relationships.json")))
    objects_dict = format_scan_dict(object_data, "objects")
    relationships_dict = format_scan_dict(relationship_data, "relationships")
    return objects_dict, relationships_dict


def get_scene_list(scene: Dict) -> (List[str], List[torch.Tensor]):
    # Returns a list of scan IDs and relative transformation matrices for an entire scene
    scan_id_set = [scene["reference"]]
    scan_tf_set = [torch.eye(4)]
    scan_changes = [[]]
    changes = []
    for follow_scan in scene["scans"]:
        scan_id_set.append(follow_scan["reference"])
        if "transform" in follow_scan.keys():
            scan_tf_set.append(torch.Tensor(follow_scan["transform"]).reshape((4, 4)).T)
        else:
            scan_tf_set.append(torch.eye(4))
        for change in follow_scan["rigid"]:
            if isinstance(change, int):
                changes.append(change)
            else:
                changes.append(change["instance_reference"])
        scan_changes.append(changes.copy())

    return scan_id_set, scan_tf_set, scan_changes


def transform_locations(node_list: List[Tuple], T: torch.Tensor) -> List[Tuple]:
    # Applies relative transforms to object positions to represent in common reference frame
    for i in range(len(node_list)):
        loc = node_list[i][1]["attributes"]["location"]
        homogenous_location = torch.reshape(torch.Tensor([loc[0], loc[1], loc[2], 1]), (4, 1))
        ref_frame_location = T @ homogenous_location
        node_list[i][1]["attributes"]["location"] = ref_frame_location[:3]

    return node_list


def format_scan_dict(unformated_dict: Dict, attribute: str) -> Dict:
    # Format raw dictionary of object nodes for all scenes
    scan_list = unformated_dict["scans"]
    formatted_dict = {}
    for scan in scan_list:
        formatted_dict[scan["scan"]] = scan[attribute]
    return formatted_dict


def format_sem_seg_dict(sem_seg_dict: Dict) -> Dict:
    object_dict = {}
    for object in sem_seg_dict["segGroups"]:
        object_dict[object["id"]] = object["obb"]["centroid"]

    return object_dict


def build_scene_graph(nodes_dict: Dict, edges_dict: Dict, scan_id: str, root: str, graph_out=False) -> (Optional, List[Tuple], List[Tuple]):
    # Returns a scene graph from raw data, including:
    #   - Nodes: objects with relevant attributes
    #   = Edges: relationships between objects

    # Extract objects in scan
    if scan_id not in nodes_dict.keys() or scan_id not in edges_dict.keys():
        return None, None, None

    # Extract position information from Semantic Segmentation results
    scan_sem_seg_file = os.path.join(root, scan_id, "semseg.v2.json")
    if os.path.isfile(scan_sem_seg_file):
        semantic_seg = json.load(open(scan_sem_seg_file))
        object_pos_list = format_sem_seg_dict(semantic_seg)
    else:
        print(f"No Semantic Segmentation File Available for {scan_id}")
        return None, None, None

    # Reformat node dictionary, include only relevant attributes, and add location
    nodes = nodes_dict[scan_id]
    input_node_list = []
    for node in nodes:
        node_copy = node.copy()
        id = int(node["id"])
        att_dict = {"label": node_copy.pop("label", None), "affordances": node_copy.pop("affordances", None),
                    "attributes": node_copy.pop("attributes", None), "global_id": node_copy.pop("global_id", None),
                    "color": node_copy.pop("ply_color", None)}

        if object_pos_list is not None:
            att_dict["attributes"]["location"] = torch.tensor(np.clip(object_pos_list[id], -100, 100)).to(torch.float32)

        att_dict["attributes"].pop("lexical", None)
        input_node_list.append((id, att_dict))

    # Extract edges from raw data
    edges = edges_dict[scan_id]

    # Can output a networkx Graph object for visualization purposes
    if graph_out:
        graph = nx.Graph()
        graph.add_nodes_from(input_node_list)
        for edge in edges:
            graph.add_edge(edge[0], edge[1])
    else:
        graph = None

    return graph, input_node_list, edges

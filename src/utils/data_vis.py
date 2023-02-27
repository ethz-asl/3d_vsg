from vedo import *
import torch


def visualize_graph(mesh_path, nodes, edges):
    scene_mesh = Mesh(mesh_path)

    node_locs = []
    for node in nodes:
        node_locs.append(node[1]["attributes"]["location"].unsqueeze(0))

    node_pts = Points(torch.cat(node_locs, dim=0).numpy(), r=12, c="black", alpha=1)

    unique_edges = []
    for edge in edges:
        sorted_edge = (edge[0], edge[1]) if edge[1] > edge[0] else (edge[1], edge[0])
        unique_edges.append(sorted_edge)
    unique_edges = set(unique_edges)

    edge_starts = []
    edge_ends = []
    for edge in unique_edges:
        edge1 = [node[1]["attributes"]["location"] for node in nodes if node[0] == edge[0]][0]
        edge2 = [node[1]["attributes"]["location"] for node in nodes if node[0] == edge[1]][0]
        edge_starts.append(edge1.unsqueeze(0))
        edge_ends.append(edge2.unsqueeze(0))

    edge_lines = Lines(torch.cat(edge_starts, dim=0).numpy(), torch.cat(edge_ends, dim=0).numpy())
    show(scene_mesh, node_pts, edge_lines).close()


def visualize_results(mesh_path, nodes, pred_var, true_var, type):
    variabilities = ["State", "Position", "Instance"]
    i = variabilities.index(type)

    scene_mesh = Mesh(mesh_path)

    node_locs = []
    colors = []
    for j in range(pred_var.shape[0]):
        node_locs.append(nodes[j][1]["attributes"]["location"].unsqueeze(0))
        pred = pred_var[j, i] > 0
        actual = true_var[j, i]
        if pred:
            if actual:
                colors.append((0., 1., 0.))
            else:
                colors.append((0.82, 0.46, 0.))
        else:
            if actual:
                colors.append((1., 1., 0.))
            else:
                colors.append((0., 0.68, 0.85))

    node_pts = Points(torch.cat(node_locs, dim=0).numpy(), r=12, c=colors, alpha=1)
    show(scene_mesh, node_pts).close()

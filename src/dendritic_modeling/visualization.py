
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def build_dendritic_tree_graph(model, branch_info=None, ablation_info=None):
    """
    Creates a networkx DiGraph where each node corresponds to a (layer_idx, branch_id).
    We'll store 'info' and 'ablation' as node attributes if provided.


    """
    G = nx.DiGraph()
    for i, layer in enumerate(model.branch_layers):
        out_dim = layer.branch_excitation.out_features  # or layer.output_dim
        for b_id in range(out_dim):
            node_id = (i, b_id)
            G.add_node(node_id)
            # store info
            if branch_info and i in branch_info:

                G.nodes[node_id]['info'] = branch_info[i]
            if ablation_info and i in ablation_info:
                G.nodes[node_id]['ablation'] = ablation_info[i]
            else:
                G.nodes[node_id]['ablation'] = 0

    for i in range(model.n_branch_layers - 1):
        out_dim_i = model.branch_layers[i].branch_excitation.out_features
        out_dim_next = model.branch_layers[i+1].branch_excitation.out_features
        for b1 in range(out_dim_i):
            for b2 in range(out_dim_next):
                G.add_edge((i,b1), (i+1, b2))

    if hasattr(model, 'soma_layer'):
        i_soma = model.n_branch_layers
        out_dim_soma = model.soma_layer.branch_excitation.out_features \
                       if model.soma_layer.input_excitatory else model.n_soma
        for b_id in range(out_dim_soma):
            node_id = (i_soma, b_id)
            G.add_node(node_id)
            if branch_info and i_soma in branch_info:
                G.nodes[node_id]['info'] = branch_info[i_soma]
            else:
                G.nodes[node_id]['info'] = 0
            G.nodes[node_id]['ablation'] = 0
        out_dim_last = model.branch_layers[-1].branch_excitation.out_features
        for b1 in range(out_dim_last):
            for b2 in range(out_dim_soma):
                G.add_edge((model.n_branch_layers-1,b1), (i_soma, b2))

    return G

def visualize_dendritic_tree(G, mode='info'):
    """
    G: networkx DiGraph
    mode: 'info' or 'ablation' or any node attribute.
    We'll do a simple layout, color by the chosen attribute.
    """
    pos = nx.spring_layout(G, seed=42)
    node_vals = []
    for n in G.nodes():
        val = G.nodes[n].get(mode, 0)
        node_vals.append(val)

    plt.figure(figsize=(8,6))
    nc = nx.draw_networkx_nodes(G, pos, node_color=node_vals, 
                                cmap='viridis', node_size=400)
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_labels(G, pos, font_color='white')
    plt.colorbar(nc, label=mode)
    plt.title(f"Dendritic Tree colored by {mode}")
    plt.axis('off')
    plt.show()


def plot_depth_info(branch_info):
    """
    If branch_info is {layer_idx: MI}, we can just do a line plot vs layer_idx.
    """
    layers = sorted(branch_info.keys())
    values = [branch_info[l] for l in layers]
    plt.figure()
    plt.plot(layers, values, marker='o')
    plt.xlabel("Layer index (depth)")
    plt.ylabel("Information (MI)")
    plt.title("Branch Info by Depth")
    plt.show()
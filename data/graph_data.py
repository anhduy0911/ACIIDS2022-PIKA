import torch
import torch_geometric
from torch_geometric.data import Data
import json
import pandas as pd
import networkx as nx
import config as CFG
import numpy as np

def convert_2_homogeneous_graph():
    df_pill = pd.read_csv('data/prescriptions/pill_data.dat', names= ["pill","prescription","tfidf"], sep = "\\")
    df_pill['tfidf'] = df_pill['tfidf'].astype(float)
    pill_list_unique = df_pill["pill"].unique()
    prescription_list_unique = df_pill['prescription'].unique()
    

def build_data():
    mapped_pill_idx = json.load(open('data/converted_graph/mapdict.json', 'r'))
    xs = mapped_pill_idx.keys()
    edge_index = []
    edge_weight = []
    
    mapped_name = pd.read_csv('data/converted_graph/mapped_name.csv', header=0)

    pill_edge = pd.read_csv('data/converted_graph/pill_edge.dat', names= ["pillA","pillB","tfidf"], sep = "@")
    pill_edge['pillA'] = pill_edge['pillA'].map(lambda a : mapped_name[mapped_name['pill'] == a].values[0][-1])
    pill_edge['pillB'] = pill_edge['pillB'].map(lambda a : mapped_name[mapped_name['pill'] == a].values[0][-1])
    # print(pill_edge.head(5))
    pill_edge.dropna(inplace=True)

    for x, y, w in pill_edge.values:
        # print(x)
        # print(y)
        # print(w)
        edge_index.append([mapped_pill_idx[x], mapped_pill_idx[y]])
        edge_weight.append(w)
        edge_index.append([mapped_pill_idx[y], mapped_pill_idx[x]])
        edge_weight.append(w)
    
    # print(mapped_pill_idx.values())
    # data = Data(x=xs, edge_index=torch.tensor(edge_index).t().contiguous(), edge_attr=torch.tensor(edge_weight))
    data = Data(x=torch.tensor(list(mapped_pill_idx.values()), dtype=torch.float32).unsqueeze(1), edge_index=torch.tensor(edge_index).t().contiguous(), edge_attr=torch.tensor(edge_weight).unsqueeze(1))
    # print(data)
    return data

def build_adj_matrix():
    mapped_pill_idx = json.load(open('data/converted_graph/mapdict.json', 'r'))
    mapped_name = pd.read_csv('data/converted_graph/mapped_name.csv', header=0)
    adj_matrix = np.zeros((CFG.n_class, CFG.n_class))

    pill_edge = pd.read_csv('data/converted_graph/pill_edge.dat', names= ["pillA","pillB","tfidf"], sep = "@")
    pill_edge['pillA'] = pill_edge['pillA'].map(lambda a : mapped_name[mapped_name['pill'] == a].values[0][-1])
    pill_edge['pillB'] = pill_edge['pillB'].map(lambda a : mapped_name[mapped_name['pill'] == a].values[0][-1])
    # print(pill_edge.head(5))
    pill_edge.dropna(inplace=True)

    for x, y, w in pill_edge.values:
        adj_matrix[mapped_pill_idx[x], mapped_pill_idx[y]] = w
    
    return adj_matrix

def visualize_graph():
    mapped_pill_idx = json.load(open('data/converted_graph/mapdict.json', 'r'))
    xs = mapped_pill_idx.keys()
    edge_index = []
    edge_weight = []
    
    mapped_name = pd.read_csv('data/converted_graph/mapped_name.csv', header=0)

    pill_edge = pd.read_csv('data/converted_graph/pill_edge.dat', names= ["pillA","pillB","tfidf"], sep = "@")
    pill_edge['pillA'] = pill_edge['pillA'].map(lambda a : mapped_name[mapped_name['pill'] == a].values[0][-1])
    pill_edge['pillB'] = pill_edge['pillB'].map(lambda a : mapped_name[mapped_name['pill'] == a].values[0][-1])
    # print(pill_edge.head(5))
    pill_edge.dropna(inplace=True)

    for x, y, w in pill_edge.values:
        # print(x)
        # print(y)
        # print(w)
        edge_index.append([mapped_pill_idx[x], mapped_pill_idx[y], w])
        edge_weight.append(w)
        edge_index.append([mapped_pill_idx[y], mapped_pill_idx[x], w])
        edge_weight.append(w)
        # if mapped_pill_idx[x] == 11 or mapped_pill_idx[y] == 11:
        #     print(f'{mapped_pill_idx[x]} {mapped_pill_idx[y]} {w}')
    
    edge_weight = np.array(edge_weight)
    print(f'weight max: {np.max(edge_weight)}, min: {np.min(edge_weight)}, mean: {np.mean(edge_weight)}, std: {np.std(edge_weight)}')
    # data = Data(x=xs, edge_index=torch.tensor(edge_index).t().contiguous(), edge_attr=torch.tensor(edge_weight))
    G = nx.Graph()
    G.add_weighted_edges_from(edge_index)
    
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 4]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 4]

    pos = nx.spring_layout(G, seed=911)  # positions for all nodes - seed for reproducibility
    # pos = nx.graphviz_layout(G)
    ill_pred = [1, 11, 14, 15, 20, 27, 31, 32, 36, 41, 42, 52, 56, 66, 75]
    values = [0.75 if node not in ill_pred else 1.0 for node in G.nodes()]
    # nodes
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('viridis'), node_color=values, node_size=200)
    # nx.draw_networkx_nodes(G, pos, node_size=200)
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1.2)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=0.05, alpha=0.5, edge_color="b", style="dashed"
    )

    # labels
    nx.draw_networkx_labels(G, pos, font_color='w', font_size=7, font_family="sans-serif")

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(CFG.log_dir_data + 'graph.png', dpi=500)

if __name__ == '__main__':
    # build_data()
    import matplotlib.pyplot as plt
    visualize_graph()
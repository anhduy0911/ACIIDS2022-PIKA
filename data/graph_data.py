import torch
from torch_geometric.data import Data
import json
import pandas as pd

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


if __name__ == '__main__':
    build_data()
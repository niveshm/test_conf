## find the stats of graphs in the dataset

from dataset import PathGraphDataset
import os
import sys
import pickle as pkl
sys.path.append(os.getcwd())
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

dataset = "icews14"
path_file = f"./preds/{dataset}/train_edge_paths.pkl"
edges_file = f"./preds/{dataset}/train_edges.pkl"
train_data = PathGraphDataset(path_file)

print(f"Number of graphs: {len(train_data)}")
## max number of nodes
max_nodes = 0
max_edges = 0
max_node_graph = None
max_edge_graph = None
total_nodes = 0
total_edges = 0
for i in range(len(train_data)):
    data = train_data[i]
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    total_nodes += num_nodes
    total_edges += num_edges
    if num_nodes > max_nodes:
        max_nodes = num_nodes
        max_node_graph = data
    if num_edges > max_edges:
        max_edges = num_edges
        max_edge_graph = data


    a = torch.sum(torch.unique(data.edge_index[0]) >= num_nodes)
    b = torch.sum(torch.unique(data.edge_index[1]) >= num_nodes)

    if a != 0 or b != 0:
        print(f"Graph {i} is invalid: {a} source nodes and {b} dest nodes >= {num_nodes}")
        print(f"Max source node: {torch.max(data.edge_index[0])}")
        print(f"Max dest node: {torch.max(data.edge_index[1])}")
        # breakpoint()

print(f"Max number of nodes: {max_nodes}")
print(f"Max number of edges: {max_edges}")
print(f"Avg number of nodes: {total_nodes/len(train_data)}")
print(f"Avg number of edges: {total_edges/len(train_data)}")

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=train_data.collate_fn)

for batch in tqdm(train_loader):
    # print(batch)
    # print(batch.num_graphs)
    # print(batch.batch)
    ## check it edge index are accurate
    num_nodes = batch.num_nodes
    a = torch.sum(torch.unique(batch.edge_index[0]) >= num_nodes)
    b = torch.sum(torch.unique(batch.edge_index[1]) >= num_nodes)

    if a != 0 or b != 0:
        print("something is wrong")



## save the largest graph
# breakpoint()
# with open(f"./max_node_graph.pkl", "wb") as f:
#     pkl.dump(max_node_graph, f)

# with open(f"./max_edge_graph.pkl", "wb") as f:
#     pkl.dump(max_edge_graph, f)


# with open("./processed_train_data.pkl", "wb") as f:
#     pkl.dump(train_data, f)


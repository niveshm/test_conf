import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle as pkl
from torch_geometric.data import Data, Batch
import os
import sys

from tqdm import tqdm
sys.path.append(os.getcwd())

# dataset = "icews14"
# path_file = f"./preds/{dataset}/edge_paths.pkl"

class PathGraphDataset(Dataset):
    def __init__(self, path_file, data_type, max_path_len=4):
        super(PathGraphDataset, self).__init__()
        with open(path_file, "rb") as f:
            self.edge_paths = pkl.load(f)

        self.data_type = data_type
        # with open(edges_file, "rb") as f:
        #     self.edges = pkl.load(f)
        
        self.max_path_len = max_path_len
        self.max_entity = 0
        if data_type == "test":
            self.graphs, self.inds = self.create_graph_data()
            self.inds = np.array(self.inds)
        else:
            self.graphs, self.preds = self.create_graph_data()
        
    
    def one_hot(self, idx, size):
        vec = [0] * size
        vec[idx] = 1
        return vec

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if self.data_type == "test":
            return self.graphs[idx]
        return self.graphs[idx], torch.tensor(self.preds[idx], dtype=torch.long)

    def collate_fn(self, batch):
        if self.data_type == "test":
            return Batch.from_data_list(batch)
        
        graphs, labels = zip(*batch)

        return Batch.from_data_list(graphs), torch.stack(labels)

    def build_graph(self, paths, quer_edge):
        src, q_rel, dst, q_ts, _ = quer_edge
        # src, q_rel, dst, q_ts = quer_edge
        node_id = dict()
        entity_map = dict()
        edge_index = []
        edge_rels = []
        edge_ts = []
        node_features = dict()

        node_id[src] = 0
        entity_map[src] = 0
        node_features[0] = [0, 1, 0]
        if src != dst:
            node_id[dst] = 1
            entity_map[dst] = 1
            curr_id = 2
            curr_ent_id = 2
            node_features[1] = [1, 0, 1]
        else:
            curr_id = 1
            curr_ent_id = 1
            node_features[0] = [0, 0, 0]  # src and dst are same

        for path in paths:
            quads = []
            prev_ent = src
            for i in range(1, len(path), 3):
                quads.append((prev_ent, path[i], path[i+1], path[i+2]))
                # if entity_map.get(path[i], None) is None:
                #     entity_map[path[i]] = curr_ent_id
                #     curr_ent_id += 1

                if entity_map.get(path[i+1], None) is None:
                    entity_map[path[i+1]] = curr_ent_id
                    curr_ent_id += 1
                prev_ent = path[i+1]
            # edge_index.extend(quads)

            for ind, edge in enumerate(quads):
                s, r, o, t = edge
                if ind != len(quads) - 1:
                    if (o, ind+1) not in node_id:
                        node_id[(o, ind+1)] = curr_id
                        curr_id += 1
                    if node_id[(o, ind+1)] not in node_features:
                        node_features[node_id[(o, ind+1)]] = [ind+1, len(quads) - (ind+1), entity_map[o]]
                # if  ind == 0 or ind == len(quads) - 1:    
                a = node_id[src] if ind == 0 else node_id[(s, ind)]
                b = node_id[dst] if ind == len(quads) - 1 else node_id[(o, ind+1)]
                edge_index.append([a, b])
                # if ind == 0:
                #     edge_index.append((node_id[src], node_id[(o, ind+1)]))
                # elif ind == len(quads) - 1:
                #     edge_index.append((node_id[(s, ind)], node_id[dst]))
                # else:
                #     edge_index.append((node_id[(s, ind)], node_id[(o, ind+1)]))
                edge_rels.append(r)
                edge_ts.append(t)
        
        node_feat = []
        for i in range(len(node_id)):
            node_feat.append(np.concatenate((self.one_hot(node_features[i][0], self.max_path_len), self.one_hot(node_features[i][1], self.max_path_len)), axis=0))
            # self.one_hot(node_features[i][2], len(entity_map))
        
        node_entities = torch.zeros(len(node_id), dtype=torch.long)
        for k, v in node_id.items():
            if type(k) == tuple:
                node_entities[v] = entity_map[k[0]]
            else:
                node_entities[v] = entity_map[k]

        self.max_entity = max(self.max_entity, max(entity_map.values()) + 1)
        edge_index = np.array(edge_index).T

        node_feat = np.array(node_feat)
        node_feat = torch.tensor(node_feat, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)#torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_rels = torch.tensor(edge_rels, dtype=torch.long)
        edge_ts = torch.tensor(edge_ts, dtype=torch.long)
        # node_entities = torch.tensor(node_entities, dtype=torch.long)

        return Data(x=node_feat, edge_index=edge_index, edge_type=edge_rels, edge_ts=edge_ts, node_ent=node_entities, src=torch.tensor(node_id[src], dtype=torch.long), dst=torch.tensor(node_id[dst], dtype=torch.long), q_rel=torch.tensor(q_rel, dtype=torch.long), q_ts=torch.tensor(q_ts, dtype=torch.long), num_edges=edge_index.size(1))


    def create_graph_data(self):
        graphs = []
        preds = []

        for edge, paths in tqdm(self.edge_paths.items()):
            all_paths = []
            for length, pths in paths.items():
                all_paths.extend(pths)
            graph = self.build_graph(all_paths, edge)
            graphs.append(graph)
            preds.append(edge[-1])

        return graphs, preds


if __name__ == "__main__":
    dataset = "icews14"
    data_type = "valid"
    path_file = f"./preds/{dataset}/{data_type}_edge_paths.pkl"
    # edges_file = f"./preds/{dataset}/edges.pkl"
    train_data = PathGraphDataset(path_file, data_type=data_type)

    ## dataloader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=train_data.collate_fn)
    for batch in train_loader:
        print(batch)
        breakpoint()

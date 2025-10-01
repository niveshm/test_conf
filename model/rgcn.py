import torch
from torch import nn
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
sys.path.append(os.getcwd())

from path_graph.dataset import PathGraphDataset


def convert_global_ind(ind, ptr, batch_ind):
    return ind + ptr[batch_ind]

class RGCN(nn.Module):
    def __init__(self, hidden_channels, num_rel, num_node_features, max_entities, num_bases, device, num_classes=2):
        super(RGCN, self).__init__()

        self.rel_emb = nn.Embedding(num_rel, hidden_channels, sparse=False)
        self.max_entities = max_entities
        self.conv1 = RGCNConv(num_node_features+max_entities, hidden_channels, num_rel, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_rel, num_bases=num_bases)
        self.conv3 = RGCNConv(hidden_channels, hidden_channels, num_rel, num_bases=num_bases)
        self.conv4 = RGCNConv(hidden_channels, hidden_channels, num_rel, num_bases=num_bases)
        self.lin = nn.Linear(2*hidden_channels, num_classes)
        self.device = device

    
    def forward(self, data, drop_prob=0.5):

        x = data.x
        grp_x = F.one_hot(data.node_ent, self.max_entities).to(self.device).to(torch.float)
        x = torch.cat((x, grp_x), dim=-1)

        x = self.conv1(x, data.edge_index, data.edge_type)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index, data.edge_type)
        x = F.relu(x)
        x = self.conv3(x, data.edge_index, data.edge_type)
        x = F.relu(x)
        x = self.conv4(x, data.edge_index, data.edge_type)


        q_o = data.dst
        batch_indices = torch.arange(len(data.ptr) - 1, device=self.device)
        q_o = convert_global_ind(q_o, data.ptr[:-1], batch_indices)
        q_r = data.q_rel
        rel_emb = self.rel_emb(q_r)
        o_emb = x[q_o]

        x = torch.cat([o_emb, rel_emb], dim=1)
        x = F.dropout(x, p=drop_prob, training=self.training)
        x = self.lin(x)

        return x


if __name__ == "__main__":
    dataset = "icews14"
    path_file = f"./preds/{dataset}/valid_edge_paths.pkl"
    train_data = PathGraphDataset(path_file, data_type="valid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RGCN(hidden_channels=128, num_rel=230*2, num_node_features=8, max_entities=62, num_bases=4, device=device, num_classes=2)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=train_data.collate_fn)
    for batch in tqdm(train_loader):
        g, l = batch
        res = model(g, 0.3)

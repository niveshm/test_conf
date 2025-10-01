import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.getcwd())
from tqdm import tqdm

from model.time_encoder import TimeEncode
from path_graph.dataset import PathGraphDataset

class MessagePassing(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(MessagePassing, self).__init__()
        self.fc = nn.Linear(input_dim, out_dim)
        self.rel_t_trans = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, edges, rel_emb, time_emb):
        # edges: [src, dst, rel, ts]
        # breakpoint()
        try:
            src, dst, rel, ts = edges[:,0], edges[:,1], edges[:,2], edges[:,3]
            rel_e = rel_emb(rel)  # E*D
            time_e = time_emb[ts]  # E*D
            rel_e = self.lrelu(self.rel_t_trans(torch.cat((rel_e, time_e), dim=-1)))
            # breakpoint()
            m = self.lrelu(self.fc(torch.cat((x[src], rel_e, x[dst]), dim=-1)))  # E**D
        except Exception as e:
            print(e)
            breakpoint()
        return m


def convert_global_ind(ind, ptr, batch_ind):
    return ind + ptr[batch_ind]

class LogicalGNN(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_rel, num_ts, max_entities, dropout, max_len, device):
        super(LogicalGNN, self).__init__()

        self.rel_emb = nn.Embedding(num_rel, hidden_dim)
        self.node_grp_emb = nn.Embedding(max_entities, hidden_dim)
        self.time_encoder = TimeEncode(hidden_dim)
        self.num_ts = num_ts
        self.device = device
        self.max_len = max_len

        self.node_emb = nn.Linear(feat_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.time_emb = self.gen_time_emb()
        self.mp = MessagePassing(input_dim=6*hidden_dim,  hidden_dim=hidden_dim, out_dim=2*hidden_dim)

        self.pred = nn.Linear(6*hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(0.2)

    def gen_time_emb(self):
        time = torch.arange(self.num_ts).to(self.device)
        time_emd = self.time_encoder(time)
        return time_emd

    def get_neighbors_dst(self, edge_index, edge_type, edge_ts, dst):
        # Get all edges going to the destination nodes
        # dst is an array
        next_edges_final = torch.tensor([], dtype=torch.long).to(self.device)
        for i in range(len(dst)):
            mask = edge_index[1] == dst[i]
            if mask.sum() == 0:
                continue
            res = torch.stack([edge_index[0, mask], edge_index[1, mask], edge_type[mask], edge_ts[mask]], dim=0)
            next_edges_final = torch.cat([next_edges_final, res], dim=1)

        return next_edges_final.T

    def get_neighbors(self, batch_ind, curr_node, edge_index, edge_type, edge_ts, dst):
        next_edges_final = torch.tensor([], dtype=torch.long).to(self.device)
        for i in range(len(batch_ind)):
            next_edges = edge_index[:, edge_index[0] == curr_node[i]][1, :]
            next_rels = edge_type[edge_index[0] == curr_node[i]]
            next_tss = edge_ts[edge_index[0] == curr_node[i]]

            # filter edges going to dst
            mask = next_edges != dst[batch_ind[i]]
            next_edges = next_edges[mask]
            next_rels = next_rels[mask]
            next_tss = next_tss[mask]

            if next_edges.size(0) > 0:
                res = torch.stack([torch.tensor([curr_node[i]]*next_edges.size(0)).to(self.device), next_edges, next_rels, next_tss, torch.tensor([batch_ind[i]]*next_edges.size(0)).to(self.device)], dim=0)
                next_edges_final = torch.cat([next_edges_final, res], dim=1)
        return next_edges_final.T

    # def get_neighbors(self, batch_ind, curr_node, edge_index, edge_ptr, edge_type, edge_ts):
    #     next_edges_final = torch.tensor([], dtype=torch.long).to(self.device)
    #     for i in range(len(batch_ind)):
    #         curr_edges = edge_index[:, edge_ptr[batch_ind[i]]:edge_ptr[batch_ind[i]]]
    #         next_edges = curr_edges[:, curr_edges[0] == curr_node[i]][1, :]
    #         next_rels = edge_type[edge_ptr[batch_ind[i]]:edge_ptr[batch_ind[i]+1]]
    #         next_rels = next_rels[curr_edges[0] == curr_node[i]]
    #         next_rels = next_rels[next_edges != 1]
    #         next_tss = edge_ts[edge_ptr[batch_ind[i]]:edge_ptr[batch_ind[i]+1]]
    #         next_tss = next_tss[curr_edges[0] == curr_node[i]]
    #         next_tss = next_tss[next_edges != 1]
    #         next_edges = next_edges[next_edges != 1]
    #         # stack edge_idx and rel and ts and batch_ind
    #         breakpoint()
    #         if next_edges.size(0) > 0:
    #             res = torch.stack([torch.tensor([curr_node[i]]*next_edges.size(0)).to(self.device), next_edges, next_rels, next_tss, torch.tensor([batch_ind[i]]*next_edges.size(0)).to(self.device)], dim=0)
    #             next_edges_final = torch.cat([next_edges_final, res], dim=1)

    #         breakpoint()
    #     pass

    # def create_edge_ptr(self, num_edges):
    #     ptr = [0]
    #     for n in num_edges:
    #         ptr.append(ptr[-1] + n)
    #     return torch.tensor(ptr, dtype=torch.long).to(self.device)

    def get_batch_indices(self, node_ids, ptr):

        batch_ind = torch.zeros_like(node_ids).to(self.device)
        for i in range(len(ptr) - 1):
            mask = (node_ids >= ptr[i]) & (node_ids < ptr[i+1])
            batch_ind[mask] = i
        return batch_ind
        

    def forward(self, batch):
        q_s = batch.src
        q_o = batch.dst
        q_r = batch.q_rel
        q_t = batch.q_ts
        # print(torch.unique(batch.edge_type))

        q_s = convert_global_ind(q_s, batch.ptr[:-1], torch.arange(len(batch.ptr) - 1).to(self.device))
        q_o = convert_global_ind(q_o, batch.ptr[:-1], torch.arange(len(batch.ptr) - 1).to(self.device))

        # edge_ptr = self.create_edge_ptr(batch.num_edges)

        x = batch.x
        ent = batch.node_ent
        ent_emb = self.node_grp_emb(ent)
        x = self.lrelu(self.node_emb(x))
        x = torch.cat([x, ent_emb], dim=-1)
        x = self.dropout(x)

        ## fetch batch for edges

        batch_ind = torch.arange(len(batch.ptr) - 1)
        # curr_x = x[q_s]#x[convert_global_ind(q_s, batch.ptr[:-1], batch_ind)]
        curr_node = q_s
        while curr_node.size(0) > 0:
            next_edges = self.get_neighbors(batch_ind, curr_node, batch.edge_index, batch.edge_type, batch.edge_ts, dst=q_o)
            if next_edges.size(0) == 0:
                break

            mess = self.mp(x, next_edges[:, :-1], self.rel_emb, self.time_emb)
            # breakpoint()

            # avg messages for each dst node
            if mess.size(0) == 0:
                break
            dst_nodes = next_edges[:,1]
            unique_dst = torch.unique(dst_nodes)
            new_x = torch.zeros_like(x).to(self.device)
            for node in unique_dst:
                # breakpoint()
                mask = dst_nodes == node
                new_x[node] = torch.mean(mess[mask], dim=0)
            
            new_x = self.dropout(new_x)
            x = x + new_x  # residual connection
            curr_node = unique_dst
            batch_ind = self.get_batch_indices(curr_node, batch.ptr)
            # breakpoint()
        
        next_edges = self.get_neighbors_dst(batch.edge_index, batch.edge_type, batch.edge_ts, dst=q_o)
        final_mess = self.mp(x, next_edges, self.rel_emb, self.time_emb)
        # batch mean
        dst_nodes = next_edges[:,1]
        new_x = torch.zeros_like(x).to(self.device)
        for i in range(q_o.size(0)):
            # breakpoint()
            mask = dst_nodes == q_o[i]
            new_x[q_o[i]] = torch.mean(final_mess[mask], dim=0)

        new_x = self.dropout(new_x)
        x = x + new_x
    

        q_r_t = self.lrelu(self.mp.rel_t_trans(torch.cat((self.rel_emb(q_r), self.time_emb[q_t]), dim=-1)))
        # breakpoint()
        # torch.cat((x[q_s], self.rel_emb(q_r), self.time_emb[q_t], x[q_o]), dim=-1)
        final_feat = torch.cat([x[q_s], q_r_t, x[q_o]], dim=-1)
        final_feat = self.dropout(final_feat)

        final_out = self.sigmoid(self.pred(final_feat))

        return final_out.squeeze(-1)


if __name__ == "__main__":
    dataset = "icews14"
    path_file = f"./preds/{dataset}/edge_paths.pkl"
    edges_file = f"./preds/{dataset}/edges.pkl"
    train_data = PathGraphDataset(path_file, edges_file)
    model = LogicalGNN(feat_dim=8, hidden_dim=128, num_rel=230*2, num_ts=364, max_entities=train_data.max_entity, dropout=0.5, max_len=3, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    ## dataloader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=train_data.collate_fn)
    for batch in tqdm(train_loader):
        res = model(batch)
        # print(res.shape)
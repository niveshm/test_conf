import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.getcwd())
from tqdm import tqdm
from torch_scatter import scatter_mean
import pickle as pkl

from model.time_encoder import TimeEncode
from path_graph.dataset import PathGraphDataset

class MessagePassing(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(MessagePassing, self).__init__()
        self.fc = nn.Linear(input_dim, out_dim)
        self.rel_t_trans = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)  # Memory efficient inplace operation

    def forward(self, x, edges, rel_emb, time_emb):
        # edges: [src, dst, rel, ts]
        if edges.size(0) == 0:
            return torch.empty(0, self.fc.out_features, device=x.device)
            
        src, dst, rel, ts = edges[:,0], edges[:,1], edges[:,2], edges[:,3]
        
        # Memory-efficient embeddings
        rel_e = rel_emb(rel)  # E*D
        time_e = time_emb[ts]  # E*D
        rel_t_combined = self.lrelu(self.rel_t_trans(torch.cat((rel_e, time_e), dim=-1)))
        
        # Efficient message computation
        src_emb = x[src]
        dst_emb = x[dst]
        message_input = torch.cat((src_emb, rel_t_combined, dst_emb), dim=-1)
        
        return self.lrelu(self.fc(message_input))


def convert_global_ind(ind, ptr, batch_ind):
    return ind + ptr[batch_ind]

class LogicalGNN(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_rel, num_ts, max_entities, dropout, max_len, device):
        super(LogicalGNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_ts = num_ts
        self.device = device
        self.max_len = max_len

        # Embeddings
        self.rel_emb = nn.Embedding(num_rel, hidden_dim)
        self.node_grp_emb = nn.Embedding(max_entities, hidden_dim)
        self.time_encoder = TimeEncode(hidden_dim)
        self.time_emb = self.gen_time_emb()
        
        # Linear layers
        self.node_emb = nn.Linear(feat_dim, hidden_dim)
        self.pred = nn.Linear(6*hidden_dim, 1)
        
        # Activations and regularization
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)  # Use inplace for memory efficiency
        
        # Message passing
        self.mp = MessagePassing(input_dim=6*hidden_dim, hidden_dim=hidden_dim, out_dim=2*hidden_dim)
        
        # Precompute time embeddings
        # self.register_buffer("time_emb", self.gen_time_emb())
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence and memory efficiency"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
    
    # def clear_cache(self):
    #     """Clear GPU cache to free memory"""
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()

    def gen_time_emb(self):
        time = torch.arange(self.num_ts, device=self.device)
        time_emd = self.time_encoder(time)
        return time_emd
    
    def aggregated_messages(self, messages, dst_nodes, num_nodes):
        """Memory-efficient message aggregation"""
        if messages.size(0) == 0:
            return torch.zeros(num_nodes, messages.size(-1), device=self.device)
        
        aggregated = scatter_mean(messages, dst_nodes, dim=0, dim_size=num_nodes)
        return aggregated

    def get_neighbors_dst(self, edge_index, edge_type, edge_ts, dst):
        """Memory-optimized version using vectorized operations"""
        # Create mask for all destination nodes at once
        dst_mask = torch.isin(edge_index[1], dst)
        
        if not dst_mask.any():
            return torch.empty(0, 4, dtype=torch.long, device=self.device)
        
        # Select all relevant edges at once without intermediate tensors
        return torch.stack([
            edge_index[0, dst_mask], 
            edge_index[1, dst_mask], 
            edge_type[dst_mask], 
            edge_ts[dst_mask]
        ], dim=1)

    # def get_neighbors_dst(self, edge_index, edge_type, edge_ts, dst):
    #     # Get all edges going to the destination nodes
    #     # dst is an array
    #     next_edges_final = torch.tensor([], dtype=torch.long).to(self.device)
    #     for i in range(len(dst)):
    #         mask = edge_index[1] == dst[i]
    #         if mask.sum() == 0:
    #             continue
    #         res = torch.stack([edge_index[0, mask], edge_index[1, mask], edge_type[mask], edge_ts[mask]], dim=0)
    #         next_edges_final = torch.cat([next_edges_final, res], dim=1)

    #     return next_edges_final.T

    def get_neighbors_optimized(self, edge_batch, curr_node, edge_index, edge_type, edge_ts, dst):
        """Memory-efficient neighbor finding with minimal tensor operations"""
        # Combine masks in one operation to reduce memory
        src_mask = torch.isin(edge_index[0], curr_node)
        dst_mask = torch.isin(edge_index[1], dst, invert=True)
        final_mask = src_mask & dst_mask
        
        if not final_mask.any():
            return torch.empty(0, 5, dtype=torch.long, device=self.device)

        # Direct tensor creation without intermediates
        return torch.stack([
            edge_index[0, final_mask],
            edge_index[1, final_mask], 
            edge_type[final_mask],
            edge_ts[final_mask],
            edge_batch[final_mask]
        ], dim=1)

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

    def create_edge_ptr(self, num_edges):
        """Memory-efficient edge pointer creation"""
        ptr = torch.zeros(len(num_edges) + 1, dtype=torch.long, device=self.device)
        ptr[1:] = torch.cumsum(torch.tensor(num_edges, device=self.device), dim=0)
        return ptr
    
    def create_edge_batch(self, num_edges, batch_edges):
        edge_batch = torch.zeros(batch_edges, dtype=torch.long, device=self.device)
        curr = 0
        for i, n_edges in enumerate(num_edges):
            edge_batch[curr:curr+n_edges] = i
            curr += n_edges
        return edge_batch


    def get_batch_indices(self, node_ids, ptr):
        batch_ind = torch.zeros_like(node_ids, device=self.device)
        for i in range(len(ptr) - 1):
            mask = (node_ids >= ptr[i]) & (node_ids < ptr[i+1])
            batch_ind[mask] = i
        return batch_ind
        

    def forward(self, batch):
        # Memory-efficient batch processing
        q_s = batch.src.to(self.device)
        q_o = batch.dst.to(self.device) 
        q_r = batch.q_rel.to(self.device)
        q_t = batch.q_ts.to(self.device)

        # Create batch indices tensor once
        batch_indices = torch.arange(len(batch.ptr) - 1, device=self.device)
        q_s = convert_global_ind(q_s, batch.ptr[:-1], batch_indices)
        q_o = convert_global_ind(q_o, batch.ptr[:-1], batch_indices)

        # Create edge batch mapping
        edge_batch = self.create_edge_batch(batch.num_edges, batch.edge_index.size(1))

        # Memory-efficient node embeddings
        x = self.lrelu(self.node_emb(batch.x.to(self.device)))
        ent_emb = self.node_grp_emb(batch.node_ent.to(self.device))
        x = torch.cat([x, ent_emb], dim=-1)
        x = self.dropout(x)
        
        # Clear intermediate variables
        del ent_emb, batch_indices

        ## fetch batch for edges

        # batch_ind = torch.arange(len(batch.ptr) - 1)
        # curr_x = x[q_s]#x[convert_global_ind(q_s, batch.ptr[:-1], batch_ind)]
        curr_node = q_s
        # Add maximum iteration limit to prevent infinite loops
        max_iterations = self.max_len
        iteration = 0
        
        while curr_node.size(0) > 0:
            next_edges = self.get_neighbors_optimized(edge_batch, curr_node, batch.edge_index, batch.edge_type, batch.edge_ts, dst=q_o)
            if next_edges.size(0) == 0:
                break

            mess = self.mp(x, next_edges[:, :-1], self.rel_emb, self.time_emb)

            # avg messages for each dst node
            if mess.size(0) == 0:
                break
            dst_nodes = next_edges[:,1]
            new_x = self.aggregated_messages(mess, dst_nodes, x.size(0))
            breakpoint()
            
            new_x = self.dropout(new_x)
            x.add_(new_x)  # In-place addition for memory efficiency
            curr_node = torch.unique(dst_nodes)
            iteration += 1
            
            # Clear intermediate variables to free memory
            del next_edges, mess, dst_nodes, new_x
        
        next_edges = self.get_neighbors_dst(batch.edge_index, batch.edge_type, batch.edge_ts, dst=q_o)
        if next_edges.size(0) > 0:  # Only process if edges exist
            final_mess = self.mp(x, next_edges, self.rel_emb, self.time_emb)
            # batch mean
            dst_nodes = next_edges[:,1]
            new_x = self.aggregated_messages(final_mess, dst_nodes, x.size(0))
            new_x = self.dropout(new_x)
            x.add_(new_x)  # In-place addition
            
            # Clear intermediate variables
            del next_edges, final_mess, dst_nodes, new_x
    

        # Final prediction with memory-efficient operations
        with torch.no_grad():
            rel_emb = self.rel_emb(q_r)
            time_emb = self.time_emb[q_t]
            
        # Enable gradients for final computation
        rel_emb = rel_emb.requires_grad_(True)
        time_emb = time_emb.requires_grad_(True)
        
        q_r_t = self.lrelu(self.mp.rel_t_trans(torch.cat((rel_emb, time_emb), dim=-1)))
        
        # Memory-efficient final feature construction
        src_feat = x[q_s]
        dst_feat = x[q_o]
        final_feat = torch.cat([src_feat, q_r_t, dst_feat], dim=-1)
        final_feat = self.dropout(final_feat)

        output = self.sigmoid(self.pred(final_feat)).squeeze(-1)
        
        # Clear intermediate variables
        del rel_emb, time_emb, q_r_t, src_feat, dst_feat, final_feat
        
        return output


if __name__ == "__main__":
    dataset = "icews14"
    path_file = f"./preds/{dataset}/valid_edge_paths.pkl"
    train_data = PathGraphDataset(path_file, data_type="valid")
    model = LogicalGNN(feat_dim=8, hidden_dim=128, num_rel=230*2, num_ts=365, max_entities=train_data.max_entity, dropout=0.5, max_len=3, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    ## dataloader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=train_data.collate_fn)
    for batch in tqdm(train_loader):
        g, l = batch
        res = model(g)
        breakpoint()
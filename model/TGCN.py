import torch
from torch import nn
from torch_geometric.data import Data


from torch_geometric.utils import softmax
from torch_scatter import scatter_add

class CompGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CompGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.attn_h = self.get_params((in_channels, 1)) # attention for head entity
        self.attn_t = self.get_params((in_channels, 1)) # attention for tail entity
        self.attn_r = self.get_params((in_channels, 1)) # attention for relation
        self.attn_ts = self.get_params((in_channels, 1)) # attention for timestamp

        self.leaky_relu = nn.LeakyReLU(0.1)

        self.w_rel = self.get_params((in_channels, out_channels))
        self.trans_w = self.get_params((in_channels, out_channels))
        self.loop_w = self.get_params((in_channels, out_channels))


    def get_params(self, dim):
        param = nn.Parameter(torch.Tensor(*dim), requires_grad=True)
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param
    
    def forward(self, g: Data, x, r, ts):
        # g: PyG object
        # x: entity embeddings [num_ent, in_channels]
        # r: relation embeddings [num_rel, in_channels]
        # ts: timestamp embeddings [num_ts, in_channels]

        h_att = x @ self.attn_h  # [num_ent, 1]
        t_att = x @ self.attn_t  # [num_ent, 1]
        r_att = r @ self.attn_r  # [num_rel, 1]
        ts_att = ts @ self.attn_ts  # [num_ts, 1]

        # print(g.edge_index.shape)
        # print(g.edge_type.shape)
        # print(g.edge_ts.shape)
        type_ids = g.edge_type
        type_e_att = r_att[type_ids]
        ts_ids = g.edge_ts
        ts_e_att = ts_att[ts_ids]

        # print(type_e_att.shape, ts_e_att.shape, h_att.shape, t_att.shape)

        tmp_h = h_att[g.edge_index[0]]
        tmp_t = t_att[g.edge_index[1]]
        alpha = self.leaky_relu(tmp_h - tmp_t + type_e_att + ts_e_att)
        # edge softmax
        # alpha = softmax(alpha, g.edge_index[0]) # change
        alpha = softmax(alpha.squeeze(-1), g.edge_index[1], num_nodes=x.size(0))
        alpha = alpha.unsqueeze(-1)


        # print(alpha.shape)
        # print(alpha[g.edge_index[0] == 0])

        ent_emb = x[g.edge_index[0]]
        rel_emb = r[type_ids]
        ts_emb = ts[ts_ids]

        edge_data = (ent_emb + ts_emb) * (rel_emb + ts_emb) # |Edges| x D
        # print(edge_data.shape)
        msg = edge_data @ self.trans_w # |Edges| x D
        msg = msg * alpha # |Edges| x D
        # print(msg.shape)
        aggr = scatter_add(msg, g.edge_index[1], dim=0, dim_size=x.size(0)) # num_ent x D
        # print(aggr.shape)

        x = x @ self.loop_w + aggr

        r = r @ self.w_rel

        return x, r

class TGCN(nn.Module):
    def __init__(self, num_ent, num_rel, input_dim, gcn_dim, n_layer, gcn_drop=0.1, act=None):
        super(TGCN, self).__init__()

        self.num_ent = num_ent
        self.num_rel = num_rel

        self.init_dim, self.gcn_dim, self.embed_dim = gcn_dim, gcn_dim, gcn_dim
        self.gcn_drop = gcn_drop
        self.act = act
        self.n_layer = n_layer

        self.conv1 = CompGCNConv(self.init_dim, self.gcn_dim)
        self.conv2 = CompGCNConv(self.gcn_dim, self.embed_dim) if n_layer == 2 else None
        self.lin_time = nn.Linear(input_dim, gcn_dim)

        self.time_ln = nn.LayerNorm(gcn_dim)
        self.ent_ln1 = nn.LayerNorm(gcn_dim)
        self.rel_ln1 = nn.LayerNorm(gcn_dim)
        self.ent_ln2 = nn.LayerNorm(gcn_dim)
        self.rel_ln2 = nn.LayerNorm(gcn_dim)

        self.drop = nn.Dropout(gcn_drop)

    def forward(self, g: Data, ent_emb, rel_emb, time_emd):
        # g: PyG object
        # ent_emb: [num_ent, init_dim]
        # rel_emb: [num_rel, init_dim]
        # time_emd: [num_ts, input_dim]

        time_emd = self.lin_time(time_emd)
        time_emd = self.time_ln(time_emd)

        x, r = ent_emb, rel_emb
        x, r = self.conv1(g, x, r, time_emd)
        x = self.ent_ln1(x)
        x = self.act(x)
        x = self.drop(x)
        r = self.rel_ln1(r)
        r = self.act(r)
        r = self.drop(r)

        if self.n_layer == 2:
            x, r = self.conv2(g, x, r, time_emd)
            x = self.ent_ln2(x)
            x = self.act(x)
            x = self.drop(x)
            r = self.rel_ln2(r)
            r = self.act(r)
            r = self.drop(r)

        return x, r
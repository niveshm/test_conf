## load best model and test
import json
import pickle as pkl
import torch
from model.logical_gnn import LogicalGNN
from path_graph.dataset import PathGraphDataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm



best_model_path = "saved_models/best_model.pth"
model_state = torch.load(best_model_path)

batch_size = 32  # Reduced from 32 to help with memory and graph issues
max_len = 3
feat_dim = 8
hidden_dim = 128
dropout = 0.5
dataset = "icews14"

test_path_file = f"./preds/{dataset}/test_edge_paths.pkl"
test_edges_file = f"./preds/{dataset}/test_edges.pkl"
test_edges = pkl.load(open(test_edges_file, "rb"))
test_data = PathGraphDataset(test_path_file, data_type="test")
print(f"Dataset loaded: {len(test_data)} samples")

rel2id = json.load(open(f"./data/{dataset}/relation2id.json", "r"))
ts2id = json.load(open(f"./data/{dataset}/ts2id.json", "r"))

num_rels = len(rel2id)*2 # for inverse relations
num_ts = len(ts2id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# breakpoint()

model = LogicalGNN(
    feat_dim=feat_dim, 
    hidden_dim=hidden_dim, 
    num_rel=num_rels, 
    num_ts=num_ts, 
    max_entities=62,
    dropout=dropout, 
    max_len=max_len, 
    device=device
)

model.load_state_dict(model_state)

test_loader = DataLoader(
    test_data, 
    batch_size=batch_size, 
    shuffle=False, 
    collate_fn=test_data.collate_fn,
)

model = model.to(device)

model.eval()
all_scores = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        scores = model(batch)
        all_scores.append(scores.cpu())
all_scores = torch.cat(all_scores, dim=0)

all_test_edges = np.array(list(test_data.edge_paths.keys()))
all_candidates = dict()

for ind, edge in tqdm(enumerate(test_edges)):
    ind_mask = test_data.inds == ind
    edge_scores = all_scores[ind_mask]
    edges = all_test_edges[ind_mask]
    sorted_scores, sorted_indices = torch.sort(edge_scores, descending=True)
    sorted_edges = edges[sorted_indices.numpy()]
    all_candidates[ind] = { e[2]: sorted_scores[i].item() for i, e in enumerate(sorted_edges)}

breakpoint()

## save all cadidates
with open(f"./preds/{dataset}/test_candidates.pkl", "wb") as f:
    pkl.dump(all_candidates, f)
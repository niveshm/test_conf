import json
import pickle as pkl
import os
import sys
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())
from tqdm import tqdm

from utils.grapher import Grapher, store_edges
import preprocess.rule_application as ra

def gen_test_edges(edge, edge_idx, data, samples=12):
    s, r, o, t = edge
    preds = set(edge_preds[edge_idx].keys())
    # breakpoint()
    
    ## all paths with s r t
    possible_obj = data[(data[:,0] == s) & (data[:,1] == r) & (data[:,2] != o) & (data[:,3] == t)]
    if possible_obj.shape[0] > 0:
        possible_obj = set(possible_obj[:,2].tolist())

    inter = preds.intersection(possible_obj)
    cands = preds - inter

    cands = list(cands)
    sorted_cands = sorted(cands, key=lambda x: edge_preds[edge_idx][x], reverse=True)
    cands = sorted_cands[:samples]
    # sample from cands
    # if len(cands) > sample_size:
    #     cands = set(np.random.choice(list(cands), sample_size, replace=False))
    edges = []
    for cand in cands:
        edges.append([s, r, cand, t, edge_idx])

    return edges

def find_matching_paths(q_src, rule, edges):
    body_rels = rule["body_rels"]
    try:
        rel_edges = edges[body_rels[0]]
        if rel_edges is None:
            return pd.DataFrame()
        rel_edges = rel_edges[rel_edges[:, 0] == q_src]

        rule_walks = pd.DataFrame(
            rel_edges,  # [sub, rel, obj, ts]
            columns=["entity_0", "rel_0", "entity_1", "timestamp_0"],
            dtype=np.uint16,
        )
        # breakpoint()

    except KeyError:
        return pd.DataFrame()

    for i in range(1, len(body_rels)):
        try:
            rel_edges = edges[body_rels[i]]
            if rel_edges is None:
                return pd.DataFrame()
            rel_edges = rel_edges  # [sub, rel, obj, ts]
            rule_walks = rule_walks.merge(
                pd.DataFrame(
                    rel_edges,
                    columns=[f"entity_{i}", f"rel_{i}", f"entity_{i+1}", f"timestamp_{i}"],
                    dtype=np.uint16,
                ),
                on=[f"entity_{i}"],
                how="inner"
            )

            rule_walks = rule_walks[rule_walks[f"timestamp_{i}"] >= rule_walks[f"timestamp_{i-1}"]]


        except KeyError:
            return pd.DataFrame()

    return rule_walks


def extract_paths(c_edges):
    print("Starting Process with num of edges:", len(c_edges))
    edge_paths = dict()
    for idx, c_edge in enumerate(c_edges):
        s, r, o, t, ind = c_edge
        edges = ra.get_window_edges(data.all_idx, t, train_edges, window=window)  #(t, data.train_idx)
        if r in rules_dict:
            rules = rules_dict[r]
            paths = {}
            for lent in rule_lengths:
                paths[lent] = []

            if len(rules) > 0:
                for rule in rules:
                    if len(paths[len(rule["body_rels"])]) >= max_paths_per_len[len(rule["body_rels"])]:
                        continue

                    rule_walks = find_matching_paths(s, rule, edges)

                    if rule_walks.shape[0] > 0:
                        # rule_walks_f = rule_walks[rule_walks[rule_walks.columns[-2]] == o]
                        # if rule_walks_f.shape[0] > 0:
                        #     print("found")
                        #     breakpoint()
                        if rule["var_constraints"]:
                            rule_walks = ra.check_var_constraints(
                                rule["var_constraints"], rule_walks
                            )
                        rule_walks = rule_walks[rule_walks[rule_walks.columns[-2]] == o]
                        if rule_walks.shape[0] > 0:
                            rule_walks = rule_walks.to_numpy()
                            # remaining walks
                            walks_to_add = max_paths_per_len[len(rule["body_rels"])] - len(paths[len(rule["body_rels"])])
                            
                            try:
                                prob = np.exp(rule_walks[:, 3] - t) / np.sum(np.exp(rule_walks[:, 3] - t))
                                rule_walks = rule_walks[np.random.choice(rule_walks.shape[0], min(walks_to_add, rule_walks.shape[0]), replace=False, p=prob)]
                            except ValueError:
                                rule_walks = rule_walks[np.random.choice(rule_walks.shape[0], min(walks_to_add, rule_walks.shape[0]), replace=False)]

                            paths[len(rule["body_rels"])].extend(rule_walks.tolist())
                            # print(len(paths[len(rule["body_rels"])]))

            # print(len(paths[1]), len(paths[2]), len(paths[3]))

            if len(paths[1]) + len(paths[2]) + len(paths[3]) > 0:
                edge_paths[(s, r, o, t, ind)] = paths
        
        if idx % 1000 == 0:
            print(f"Processed {idx} edges...")
    
    return edge_paths


rules_file = "150925115904_r[1,2,3]_n200_exp_s12_rules.json"
dataset = "icews14"
data_type = "test"
rules_dict = json.load(open(f"rules/{dataset}/" + rules_file))
rules_dict = {int(k): v for k, v in rules_dict.items()}
rule_lengths = [1, 2, 3]
max_paths_per_len = {1:200, 2:100, 3:100}
neg_samples = 3
window = 0

data = Grapher(dataset, data_type=data_type)

ra.rules_statistics(rules_dict)
rules_dict = ra.filter_rules(rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=[1,2,3])
ra.rules_statistics(rules_dict)

## sort rules by confidence
for key in rules_dict.keys():
    rules_dict[key] = sorted(rules_dict[key], key=lambda x: x["conf"], reverse=True)

with open(f"./preds/{dataset}/{data_type}_edges.pkl", "rb") as f:
    training_edges = pkl.load(f)

edge_preds = json.load(open(f"./preds/{dataset}/{data_type}_cands.json"))

edge_preds = {int(k): { int(x):y for x, y in v.items() }for k, v in edge_preds.items()}
train_edges = store_edges(data.train_idx)

all_edges = []

for ind, edge in tqdm(enumerate(training_edges)):
    s, r, o, t = edge
    edges = gen_test_edges(edge, ind, data.all_idx)
    all_edges.extend(edges)


all_edges = np.array(all_edges)
print("Total edges for path extraction:", all_edges.shape[0])

num_processes = 12
num_queries = len(all_edges) // num_processes + 1
output = Parallel(n_jobs=num_processes)(
    delayed(extract_paths)(all_edges[i*num_queries : (i+1)*num_queries]) for i in range(num_processes)
)

edge_paths = dict()
for out in output:
    edge_paths.update(out)


with open(f"./preds/{dataset}/{data_type}_edge_paths.pkl", "wb") as f:
    pkl.dump(edge_paths, f)
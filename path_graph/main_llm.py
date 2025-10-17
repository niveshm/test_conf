import json
import os
import sys
import pickle as pkl
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append(os.getcwd())

from utils.grapher import Grapher, store_edges
import preprocess.rule_application as ra
from joblib import Parallel, delayed

rules_file = "150925115904_r[1,2,3]_n200_exp_s12_rules.json"
dataset = "icews14"
data_type = "valid"
rules_dict = json.load(open(f"rules/{dataset}/" + rules_file))
rules_dict = {int(k): v for k, v in rules_dict.items()}
rule_lengths = [1, 2, 3]
max_paths_per_len = {1:5, 2:5, 3:5}
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
    

    ## drop timestamp columns
    ts_cols = [col for col in rule_walks.columns if "timestamp" in col]
    rule_walks = rule_walks.drop(columns=ts_cols)

    return rule_walks


# def filter_edges(q_time, edges):
#     edges = edges[edges[:, 3] < q_time]
#     # print(edges.shape)
#     return store_edges(edges)


def extract_neg_edges(edge, edge_idx, data, sample_size=10):
    s, r, o, t = edge
    preds = set(edge_preds[edge_idx].keys())
    # breakpoint()
    if o not in preds:
        return []
    
    ## all paths with s r t
    possible_obj = data[(data[:,0] == s) & (data[:,1] == r) & (data[:,3] == t)]
    if possible_obj.shape[0] > 0:
        possible_obj = set(possible_obj[:,2].tolist())

    inter = preds.intersection(possible_obj)
    neg_cands = preds - inter

    # sort neg_cands
    neg_cands = sorted(list(neg_cands), key=lambda x: edge_preds[edge_idx][x], reverse=True)
    neg_cands = neg_cands[:sample_size]

    # sample from neg_cands
    # if len(neg_cands) > sample_size:
    #     neg_cands = set(np.random.choice(list(neg_cands), sample_size, replace=False))
    neg_edges = [[s, r, o, t, 1]]
    for cand in neg_cands:
        neg_edges.append([s, r, cand, t, 0])
    
    return neg_edges


def extract_paths(c_edges):
    print("Starting Process with num of edges:", len(c_edges))
    edge_paths = dict()
    for idx, c_edge in enumerate(c_edges):
        s, r, o, t, y = c_edge
        edges = ra.get_window_edges(data.all_idx, t, train_edges, window=window)  #(t, data.train_idx)
        if r in rules_dict:
            rules = rules_dict[r]
            paths = {}
            for lent in rule_lengths:
                paths[lent] = set()

            if len(rules) > 0:
                for rule in rules:
                    if len(paths[len(rule["body_rels"])]) >= max_paths_per_len[len(rule["body_rels"])]:
                        continue

                    rule_walks = find_matching_paths(s, rule, edges)
                    # breakpoint()

                    if rule_walks.shape[0] > 0:
                        # rule_walks_f = rule_walks[rule_walks[rule_walks.columns[-2]] == o]
                        # if rule_walks_f.shape[0] > 0:
                        #     print("found")
                        if rule["var_constraints"]:
                            rule_walks = ra.check_var_constraints(
                                rule["var_constraints"], rule_walks
                            )
                        rule_walks = rule_walks[rule_walks["entity_"+str(len(rule["body_rels"]))] == o]

                        ## find unique paths
                        
                        if rule_walks.shape[0] > 0:
                            # breakpoint()
                            rule_walks = rule_walks.drop_duplicates()
                            rule_walks = rule_walks.to_numpy()
                            # remaining walks
                            walks_to_add = max_paths_per_len[len(rule["body_rels"])] - len(paths[len(rule["body_rels"])])

                            
                            
                            # try:
                            #     prob = np.exp(rule_walks[:, 3] - t) / np.sum(np.exp(rule_walks[:, 3] - t))
                            #     rule_walks = rule_walks[np.random.choice(rule_walks.shape[0], min(walks_to_add, rule_walks.shape[0]), replace=False, p=prob)]
                            # except ValueError:
                            
                            # rule_walks = rule_walks[np.random.choice(rule_walks.shape[0], min(walks_to_add, rule_walks.shape[0]), replace=False)]
                            # breakpoint()
                            cnt = 0
                            # breakpoint()
                            for rule_walk in rule_walks.tolist():
                                if cnt >= walks_to_add:
                                    break

                                
                                ## don't add if already present
                                # if len(paths[len(rule["body_rels"])]) > 0:
                                #     if not np.any(np.all(np.array(paths[len(rule["body_rels"])]) == np.array(rule_walk), axis=1)):
                                #         paths[len(rule["body_rels"])] = np.vstack([paths[len(rule["body_rels"])], np.array(rule_walk)])
                                
                                # else:
                                #     paths[len(rule["body_rels"])] = np.array([rule_walk])
                                # cnt += 1

                                if tuple(rule_walk) not in paths[len(rule["body_rels"])]:
                                    paths[len(rule["body_rels"])].add(tuple(rule_walk))
                                    cnt += 1

                                # breakpoint()

                            # paths[len(rule["body_rels"])].extend(rule_walks.tolist())
                            # print(len(paths[len(rule["body_rels"])]))

            # print(len(paths[1]), len(paths[2]), len(paths[3]))

            for lent in rule_lengths:
                paths[lent] = list(paths[lent])

            if len(paths[1]) + len(paths[2]) + len(paths[3]) > 0:
                edge_paths[(s, r, o, t, y)] = paths
            # else:
            #     print("No paths found for edge:", (s, r, o, t, y))
        
        if idx % 1000 == 0:
            print(f"Processed {idx} edges...")
    
    return edge_paths



all_edges = set()

print("Generating negative samples...")
for ind, edge in tqdm(enumerate(training_edges)):
    s, r, o, t = edge
    neg_edges = extract_neg_edges(edge, ind, data.all_idx, sample_size=neg_samples)
    for neg_edge in neg_edges:
        ## check if neg_edge already in training edges
        # if len(all_edges) == 0:
        #     all_edges = np.array([neg_edge])
        # else:
        #     if not np.any(np.all(all_edges == neg_edge, axis=1)):
        #         all_edges = np.vstack([all_edges, neg_edge])

        if tuple(neg_edge) not in all_edges:
            all_edges.add(tuple(neg_edge))
    # all_edges = np.concatenate([all_edges, neg_edges], axis=0)

    # print(edge_preds[ind])

all_edges = np.array(list(all_edges), dtype=training_edges.dtype)
# training_edges = np.concatenate((training_edges, np.ones((training_edges.shape[0], 1), dtype=training_edges.dtype)), axis=1)
# all_edges = np.concatenate((all_neg_edges, training_edges), axis=0)
print("all edges: ", len(all_edges))
num_processes = 12
num_queries = len(all_edges) // num_processes + 1
output = Parallel(n_jobs=num_processes)(
    delayed(extract_paths)(all_edges[i*num_queries : (i+1)*num_queries]) for i in range(num_processes)
)

# output = extract_paths([(np.int64(6795), np.int64(155), np.int64(2083), np.int64(237), np.int64(1))])
# breakpoint()

edge_paths = dict()
for out in output:
    edge_paths.update(out)

    # if r in rules_dict:
    #     rules = rules_dict[r]
    #     paths = {}
    #     for lent in rule_lengths:
    #         paths[lent] = []

    #     if len(rules) > 0:
    #         for rule in rules:
    #             if len(paths[len(rule["body_rels"])]) >= max_paths_per_len[len(rule["body_rels"])]:
    #                 continue

    #             rule_walks = find_matching_paths(s, rule, edges)

    #             if rule_walks.shape[0] > 0:
    #                 # rule_walks_f = rule_walks[rule_walks[rule_walks.columns[-2]] == o]
    #                 # if rule_walks_f.shape[0] > 0:
    #                 #     print("found")
    #                 #     breakpoint()
    #                 if rule["var_constraints"]:
    #                     rule_walks = ra.check_var_constraints(
    #                         rule["var_constraints"], rule_walks
    #                     )
    #                 rule_walks = rule_walks[rule_walks[rule_walks.columns[-2]] == o]
    #                 if rule_walks.shape[0] > 0:
    #                     paths[len(rule["body_rels"])].extend(rule_walks.to_numpy().tolist())
        
    #     if len(paths[1]) + len(paths[2]) + len(paths[3]) > 0:
    #         edge_paths[(s, r, o, t, 1)] = paths

# print("Number of positive edges:", len(edge_paths))
# print("Number of negative edges:", len(all_neg_edges))
# print("Processing negative samples...")
# for edge in tqdm(all_neg_edges):
#     s, r, o, t, y = edge
#     edges = filter_edges(t, data.train_idx)
#     if r in rules_dict:
#         rules = rules_dict[r]
#         paths = {}
#         for lent in rule_lengths:
#             paths[lent] = []

#         if len(rules) > 0:
#             for rule in rules:
#                 if len(paths[len(rule["body_rels"])]) >= max_paths_per_len[len(rule["body_rels"])]:
#                     continue

#                 rule_walks = find_matching_paths(s, rule, edges)

#                 if rule_walks.shape[0] > 0:
#                     if rule["var_constraints"]:
#                         rule_walks = ra.check_var_constraints(
#                             rule["var_constraints"], rule_walks
#                         )
#                     rule_walks = rule_walks[rule_walks[rule_walks.columns[-2]] == o]
#                     if rule_walks.shape[0] > 0:
#                         paths[len(rule["body_rels"])].extend(rule_walks.to_numpy().tolist())
        
#         if len(paths[1]) + len(paths[2]) + len(paths[3]) > 0:
#             edge_paths[(s, r, o, t, 0)] = paths




## save paths
with open(f"./preds/{dataset}/{data_type}_edge_paths_notime.pkl", "wb") as f:
    pkl.dump(edge_paths, f)
            

        

                


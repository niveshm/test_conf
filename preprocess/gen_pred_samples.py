import json
import time
import argparse
import itertools
import numpy as np
from joblib import Parallel, delayed
import pickle as pkl
import os
import sys
sys.path.append(os.getcwd())

from utils.grapher import Grapher, store_edges
from score_function import score_12
import rule_application as ra


dataset = "icews14"
data_type = "train"
rules_file = "150925115904_r[1,2,3]_n200_exp_s12_rules.json"
sample_perc = 0.5
window = 0
top_k = 20
num_processes = 8
rule_lengths = [1, 2, 3]
data = Grapher(dataset, data_type=data_type)
test = Grapher(dataset, data_type=data_type, sample_perc=sample_perc)
rules_dict = json.load(open(f"./rules/{dataset}/" + rules_file))
rules_dict = {int(k): v for k, v in rules_dict.items()}
print("Rules statistics:")
ra.rules_statistics(rules_dict)
rules_dict = ra.filter_rules(
    rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=rule_lengths
)
ra.rules_statistics(rules_dict)
if data_type == "valid":
    test_data = test.valid_idx
elif data_type == "test":
    test_data = test.test_idx
else:
    test_data = test.train_idx
print("Number of test queries:", len(test_data))

learn_edges = store_edges(data.train_idx)
score_func = score_12
print("Test data length: ", len(test_data))

args = [[0.1, 0.5]]

def apply_rules(i, num_queries):
    """
    Apply rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_queries (int): minimum number of queries for each process

    Returns:
        all_candidates (list): answer candidates with corresponding confidence scores
        no_cands_counter (int): number of queries with no answer candidates
    """

    print("Start process", i, "...")
    all_candidates = [dict() for _ in range(len(args))]
    no_cands_counter = 0

    num_rest_queries = len(test_data) - (i + 1) * num_queries
    if num_rest_queries >= num_queries:
        test_queries_idx = range(i * num_queries, (i + 1) * num_queries)
    else:
        test_queries_idx = range(i * num_queries, len(test_data))

    cur_ts = test_data[test_queries_idx[0]][3]
    edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)

    it_start = time.time()
    for j in test_queries_idx:
        test_query = test_data[j]
        cands_dict = [dict() for _ in range(len(args))]

        if test_query[3] != cur_ts:
            cur_ts = test_query[3]
            edges = ra.get_window_edges(data.all_idx, cur_ts, learn_edges, window)

        if test_query[1] in rules_dict:
            dicts_idx = list(range(len(args)))
            for rule in rules_dict[test_query[1]]:
                walk_edges = ra.match_body_relations(rule, edges, test_query[0])

                if 0 not in [len(x) for x in walk_edges]:
                    rule_walks = ra.get_walks(rule, walk_edges)
                    if rule["var_constraints"]:
                        rule_walks = ra.check_var_constraints(
                            rule["var_constraints"], rule_walks
                        )

                    if not rule_walks.empty:
                        cands_dict = ra.get_candidates(
                            rule,
                            rule_walks,
                            cur_ts,
                            cands_dict,
                            score_func,
                            args,
                            dicts_idx,
                        )
                        for s in dicts_idx:
                            cands_dict[s] = {
                                x: sorted(cands_dict[s][x], reverse=True)
                                for x in cands_dict[s].keys()
                            }
                            cands_dict[s] = dict(
                                sorted(
                                    cands_dict[s].items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            )
                            top_k_scores = [v for _, v in cands_dict[s].items()][:top_k]
                            unique_scores = list(
                                scores for scores, _ in itertools.groupby(top_k_scores)
                            )
                            if len(unique_scores) >= top_k:
                                dicts_idx.remove(s)
                        if not dicts_idx:
                            break

            if cands_dict[0]:
                for s in range(len(args)):
                    # Calculate noisy-or scores
                    scores = list(
                        map(
                            lambda x: 1 - np.prod(1 - np.array(x)),
                            cands_dict[s].values(),
                        )
                    )
                    cands_scores = dict(zip(cands_dict[s].keys(), scores))
                    noisy_or_cands = dict(
                        sorted(cands_scores.items(), key=lambda x: x[1], reverse=True)
                    )
                    all_candidates[s][j] = noisy_or_cands
            else:  # No candidates found by applying rules
                no_cands_counter += 1
                for s in range(len(args)):
                    all_candidates[s][j] = dict()

        else:  # No rules exist for this relation
            no_cands_counter += 1
            for s in range(len(args)):
                all_candidates[s][j] = dict()

        if not (j - test_queries_idx[0] + 1) % 100:
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            print(
                "Process {0}: test samples finished: {1}/{2}, {3} sec".format(
                    i, j - test_queries_idx[0] + 1, len(test_queries_idx), it_time
                )
            )
            it_start = time.time()

    return all_candidates, no_cands_counter


start = time.time()
## print current directory files
num_queries = len(test_data) // num_processes
output = Parallel(n_jobs=num_processes)(
    delayed(apply_rules)(i, num_queries) for i in range(num_processes)
)
end = time.time()

final_all_candidates = [dict() for _ in range(len(args))]
for s in range(len(args)):
    for i in range(num_processes):
        final_all_candidates[s].update(output[i][0][s])
        output[i][0][s].clear()

final_no_cands_counter = 0
for i in range(num_processes):
    final_no_cands_counter += output[i][1]

total_time = round(end - start, 6)
print("Application finished in {} seconds.".format(total_time))
print("No candidates: ", final_no_cands_counter, " queries")


dir_path = f"./preds/{dataset}/"
for s in range(len(args)):
    score_func_str = score_func.__name__ + str(args[s])
    score_func_str = score_func_str.replace(" ", "")
    ra.save_candidates(
        data_type,
        dir_path,
        final_all_candidates[s],
    )
    
    with open(f"{dir_path}/{data_type}_edges.pkl", "wb") as f:
        pkl.dump(test_data, f)

import json
import pickle as pkl
from tqdm import tqdm
import torch

dataset = "icews14"
data_type = "valid"
path_file = f"./preds/{dataset}/{data_type}_edge_paths.pkl"

with open(path_file, "rb") as f:
    edge_paths = pkl.load(f)


entity2id = json.load(open(f"./data/{dataset}/entity2id.json", "rb"))
id2entity = {v:k.replace("_", " ") for k, v in entity2id.items()}
ts2id = json.load(open(f"./data/{dataset}/ts2id.json", "rb"))
id2ts = {v:k for k, v in ts2id.items()}
relation2id_old = json.load(open(f"./data/{dataset}/relation2id.json", "rb"))
relation2id = relation2id_old.copy()
counter = len(relation2id_old)
for relation in relation2id_old:
    relation2id[relation + " (inverse)"] = counter  # Inverse relation
    counter += 1

id2relation = {v:k.replace("_", " ") for k, v in relation2id.items()}

def create_query(query_edge, paths, y):
    # print(len(paths))
    h, r, t, ts = query_edge
    h = id2entity[h]
    r = id2relation[r]
    t = id2entity[t]
    ts = id2ts[ts]


    ## (h, r, ?, ts) to query text
    # input_text = f"Query: At the time {ts}, what entity is connected to {h} by the relation {r}? Is the correct answer {t}?"
    input_text = f"Query: what entity is connected to '{h}' by the relation '{r}' at time '{ts}'? Is the correct answer '{t}'?"
    # return_vals = []

    ## remove \t and \n from input_text
    input_text = input_text.replace("\t", " ").replace("\n", " ")

    final_str = ""

    for path in paths:
        context_text = "Given that: "
        prev_ent = path[0]
        prev_ent = id2entity[prev_ent]#.replace("_", " ")

        for i in range(1, len(path), 3):
            rel = path[i]
            ent = path[i+1]
            ts = path[i+2]
            ent = id2entity[ent]#.replace("_", " ")
            rel = id2relation[rel]
            # if rel.startswith("_"):
            #     rel = rel[1:] + " (inverse)"
            # else:
            #     rel = rel.replace("_", " ")
            ts = id2ts[ts]

            context_text += f"entity '{prev_ent}' is connected to entity '{ent}' by the relation '{rel}' at time '{ts}'. "
            prev_ent = ent

        context_text += f""
        context_text = context_text.replace("\t", " ").replace("\n", " ")
        final_str += "\t".join([input_text, context_text, str(y)]) + "\n"
        # return_vals.append({
        #     "input_text": input_text,
        #     "context_text": context_text,
        #     "label": torch.tensor(y, dtype=torch.long)
        # })

    return final_str

query_data = []
    # preds = []

for edge, paths in tqdm(edge_paths.items()):
    for path in paths.values():
        if data_type != "test":
            h, r, t, ts, y = edge
        else:
            h, r, t, ts = edge
            y = -1  # dummy label for test set
        query = create_query((h,r,t,ts), path, y)
        query_data.append(query)
        # query_data = "\n".join([query_data, query])
        

query_data = "".join(query_data)


## save query_data to a text file
with open(f"./preds/{dataset}/{data_type}_llm_data.txt", "w") as f:
    f.write(query_data)

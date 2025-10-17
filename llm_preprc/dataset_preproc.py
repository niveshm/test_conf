from torch.utils.data import Dataset
import pickle as pkl
import json
import torch
from tqdm import tqdm


class PathTextDataset_prep(Dataset):
    def __init__(self, path_file, dataset, data_type="train"):
        super(PathTextDataset_prep, self).__init__()
        with open(path_file, "rb") as f:
            self.edge_paths = pkl.load(f)
        
        entity2id = json.load(open(f"./data/{dataset}/entity2id.json", "rb"))
        self.id2entity = {v:k.replace("_", " ") for k, v in entity2id.items()}
        self.ts2id = json.load(open(f"./data/{dataset}/ts2id.json", "rb"))
        self.id2ts = {v:k for k, v in self.ts2id.items()}
        self.relation2id_old = json.load(open(f"./data/{dataset}/relation2id.json", "rb"))
        self.relation2id = self.relation2id_old.copy()
        counter = len(self.relation2id_old)
        for relation in self.relation2id_old:
            self.relation2id[relation + " (inverse)"] = counter  # Inverse relation
            counter += 1

        self.id2relation = {v:k.replace("_", " ") for k, v in self.relation2id.items()}
        self.data_type = data_type

        self.data = self.process_data()

    
    def create_query(self, query_edge, paths, y):
        # print(len(paths))
        h, r, t, ts = query_edge
        h = self.id2entity[h]
        r = self.id2relation[r]
        t = self.id2entity[t]
        ts = self.id2ts[ts]


        ## (h, r, ?, ts) to query text
        # input_text = f"Query: At the time {ts}, what entity is connected to {h} by the relation {r}? Is the correct answer {t}?"
        input_text = f"Query: what entity is connected to {h} by the relation {r} at time {ts}? Is the correct answer {t}?"
        return_vals = []

        for path in paths:
            context_text = "Given that: "
            prev_ent = path[0]
            prev_ent = self.id2entity[prev_ent]#.replace("_", " ")

            for i in range(1, len(path), 3):
                rel = path[i]
                ent = path[i+1]
                ts = path[i+2]
                # ent = self.id2entity[ent].replace("_", " ")
                rel = self.id2relation[rel]
                # if rel.startswith("_"):
                #     rel = rel[1:] + " (inverse)"
                # else:
                #     rel = rel.replace("_", " ")
                ts = self.id2ts[ts]

                context_text += f"{prev_ent} is connected to {ent} by the relation {rel} at time {ts}. "
                prev_ent = ent

            context_text += f""
            return_vals.append({
                "input_text": input_text,
                "context_text": context_text,
                "label": torch.tensor(y, dtype=torch.long)
            })

        return return_vals

    def process_data(self):
        query_data = []
        # preds = []

        for edge, paths in tqdm(self.edge_paths.items()):
            for path in paths.values():
                if self.data_type != "test":
                    h, r, t, ts, y = edge
                else:
                    h, r, t, ts = edge
                    y = -1  # dummy label for test set
                query = self.create_query((h,r,t,ts), path, y)
                query_data.append(query)

        return query_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.data_type == "test":
            return self.data[idx]["input_text"], self.data[idx]["context_text"] #, self.data[idx]["label"]
        return self.data[idx]["input_text"], self.data[idx]["context_text"], self.data[idx]["label"]
    
    



if __name__ == "__main__":
    dataset = PathTextDataset_prep("./preds/icews14/train_edge_paths.pkl", "icews14", data_type="train")
    print(len(dataset))
    for i in range(10):
        print(dataset[i])
    breakpoint()
    
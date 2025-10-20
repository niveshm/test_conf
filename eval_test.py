import torch
from model.bert import ConfPredict
from torch.utils.data import DataLoader
from llm_preprc.test_dataset import TestPathTextDataset
from tqdm import tqdm
import pickle as pkl
import numpy as np
import os


model = ConfPredict()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dict = torch.load(f"./saved_models/best_bert_model.pth", map_location=device)
model.load_state_dict(model_dict)
model = model.to(device)

num_processes = 12

dataset = "icews14"
batch_size = 32
test_edges_file = f"./preds/{dataset}/test_edges.pkl"
test_edges = pkl.load(open(test_edges_file, "rb"))
a = 0.5
lmda = 0.1
preds_file = f"./preds/{dataset}/bert_lmda{lmda}_a{a}_preds.pkl"
org_preds_file = f"./preds/{dataset}/bert_lmda{lmda}_a{a}_org_preds.pkl"
preds = {}

## check if preds_file exists
# if os.path.exists(preds_file):
#     preds = pkl.load(open(preds_file, "rb"))
#     print(f"Loaded existing predictions from {preds_file}")


def get_scores(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            # ind, cand, max_ts, squeezed_encoding
            inds, cands, max_tss, encodings = batch
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            token_type_ids = encodings['token_type_ids'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            for i in range(len(inds)):
                ts = test_edges[int(inds[i])][-1]
                # np.exp(
                #     lmda * (max_cands_ts - test_query_ts)
                # )
                score = a * probs[i] + (1-a) * np.exp(lmda * (max_tss[i].item() - ts))
                # breakpoint()
                try:
                    preds[inds[i].item()][cands[i].item()].append(score)
                except KeyError:
                    # breakpoint()
                    if inds[i].item() not in preds:
                        preds[inds[i].item()] = {}
                    preds[inds[i].item()][cands[i].item()] = [score]
                
                # breakpoint()

for process_id in range(num_processes):
    text_data_path = f"./preds/{dataset}/process_{process_id}_edges.txt"

    test_dataset = TestPathTextDataset(text_data_path, max_length=250)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    get_scores(model, test_loader, device)


## save preds
with open(org_preds_file, "wb") as f:
    pkl.dump(preds, f)

for ind in preds:
    for cand in preds[ind]:
        preds[ind][cand] = 1 - np.prod(1 - np.array(preds[ind][cand]))
    preds[ind] = dict(sorted(preds[ind].items(), key=lambda item: item[1], reverse=True))
with open(preds_file, "wb") as f:
    pkl.dump(preds, f)
print(f"Saved predictions to {preds_file}")
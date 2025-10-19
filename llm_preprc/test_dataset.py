import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import sys

sys.path.append(os.getcwd())

class TestPathTextDataset(Dataset):
    def __init__(self, text_file, max_length=512):
        super(TestPathTextDataset, self).__init__()
        with open(text_file, "r") as f:
            self.data = f.readlines()
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        parts = self.data[idx].strip("\n").split("\t")

        # if len(parts) != 2:
        #     raise ValueError(f"Expected 2 parts per line, got {len(parts)} parts.")

        ind, cand, max_ts, que, con  = parts

        input_text = f"[CLS] {que} [SEP] {con} [SEP]"
        input_text = self.tokenizer(
            que,
            con,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        squeezed_encoding = {key: val.squeeze(0) for key, val in input_text.items()}

        return int(ind), int(cand), int(max_ts), squeezed_encoding



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = TestPathTextDataset("./preds/icews14/process_0_edges.txt", max_length=250)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    print(len(dataset))
    breakpoint()
    for batch in dataloader:
        print(batch)
        breakpoint()
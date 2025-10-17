from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer


class PathTextDataset(Dataset):
    def __init__(self, text_file, data_type="train", max_length=512):
        super(PathTextDataset, self).__init__()
        with open(text_file, "r") as f:
            self.data = f.readlines()
        self.data_type = data_type
        if data_type == "train":
            self.data = self.data[:700000]
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

        # self.data = self.process_data()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        parts = self.data[idx].strip("\n").split("\t")

        if len(parts) != 3:
            raise ValueError(f"Expected 3 parts per line, got {len(parts)} parts.")

        que, con, lab = parts

        input_text = f"[CLS] {que} [SEP] {con} [SEP]"
        # input_text = self.tokenizer.encode_plus(
        #     input_text,
        #     max_length=self.max_length,
        #     add_special_tokens=False,
        #     return_token_type_ids=True,
        #     # pad_to_max_length=True,
        #     padding="max_length",
        #     return_attention_mask=True,
        #     return_tensors="pt",
        # )
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

        if self.data_type != "test":
            return squeezed_encoding, torch.tensor(int(lab), dtype=torch.long)
        return squeezed_encoding
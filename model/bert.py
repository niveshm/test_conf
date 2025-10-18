import torch
import torch.nn as nn
from transformers import BertModel #, AutoTokenizer


class ConfPredict(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=2, device="cpu"):
        super(ConfPredict, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert = self.bert.to(device)
        # self.device = device
        d_model = self.bert.pooler.dense.out_features
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model*2, num_classes)
        )

        # self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        # input_ids = x["input_ids"].to(self.device)
        # attn_mask = x["attention_mask"].to(self.device)

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        pooled_output = output.pooler_output
        logits = self.classifier(pooled_output)
        # probs = self.softmax(logits)
        return logits
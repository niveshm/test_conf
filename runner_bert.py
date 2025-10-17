import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
from tqdm import tqdm


from llm_preprc.dataset import PathTextDataset
from model.bert import ConfPredict


def train_model(model, dataloader, criterion, optimizer, scheduler, device):
    model = model.to(device).train()
    criterion = criterion.to(device)

    for i, batch in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        input_txt, label = batch
        labels = label.to(device)
        # input_txt = input_txt.to(device)
        input_txt = {key: val.to(device) for key, val in input_txt.items()}
        # breakpoint()
        # input_txt = batch["input_txt"]
        # labels = batch["labels"].to(device)
        output = model(**input_txt)

        loss = criterion(output, labels)

        if i % 500 == 0:
            # with torch.no_grad():
            #     diff = output - labels
            #     abs_diff = torch.abs(diff)
            #     mae = torch.mean(abs_diff)
            # print(f"{i}\t MSE:{loss.item()}\t MAE:{mae}")
            print(f"CE loss {loss.item()}")
        #     pass
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()


def test_model(model, dataloader, criterion, device):
    with torch.no_grad():
        all_outputs = []
        all_labels = []
        model = model.to(device).eval()
        for i, batch in tqdm(enumerate(dataloader)):
            input_txt, labels = batch
            labels = labels.to(device)
            # input_txt = input_txt.to(device)
            input_txt = {key: val.to(device) for key, val in input_txt.items()}
            output = model(**input_txt)
            all_outputs.append(torch.argmax(output, dim=1))
            all_labels.append(labels)
            # diff = output_list - label_list
            # abs_diff = torch.abs(diff)
            # mae = torch.mean(abs_diff)
            # mse = criterion(output_list, label_list).item()
        # print(f"{mode} \t MSE: {mse}\t MAE: {mae}")
        output_list = torch.cat(all_outputs, dim=0)
        label_list = torch.cat(all_labels, dim=0)
        print(f"Accuracy: {torch.sum(output_list == label_list).item() / label_list.numel()}")
        # return accuracy
        return torch.sum(output_list == label_list).item() / label_list.numel()


def main():
    
    data_path = "./preds/icews14/"
    batch_size = 32
    max_length = 250
    num_epochs = 5
    train_dataset = PathTextDataset(data_path + "train_llm_data_notime.txt", "train", max_length)
    valid_dataset = PathTextDataset(data_path + "valid_llm_data_notime.txt", "valid", max_length)
    print("created datasets")

    # max_len = 0
    # for data in tqdm(train_dataset):
    #     if len(data[0]['input_ids']) > max_len:
    #         max_len = len(data[0]['input_ids'])
    
    # print("max len: ", max_len)
    # for data in tqdm(valid_dataset):
    #     if len(data[0]['input_ids']) > max_len:
    #         max_len = len(data[0]['input_ids'])
    
    # print("max len: ", max_len)
    

    # breakpoint()

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConfPredict()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    print("created model and criterion")

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)
    print("created optimizer and scheduler")

    best_acc = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        # train_model(model, train_dataloader, criterion, optimizer, scheduler, device)
        acc = test_model(model, valid_dataloader, criterion, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "./saved_models/best_bert_model.pth")
            print("Best model saved.")


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()
    
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from model.logical_gnn import LogicalGNN
from path_graph.dataset import PathGraphDataset

def train(model, train_loader, valid_loader, optimizer, device, loss_fn, num_epochs):
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            graphs, labels = batch
            graphs = graphs.to(device)
            labels = labels.float().to(device)
            out = model(graphs)
            loss = loss_fn(out, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        model.eval()
        valid_loss = 0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                graphs, labels = batch
                graphs = graphs.to(device)
                labels = labels.float().to(device)
                out = model(graphs)
                loss = loss_fn(out, labels)
                valid_loss += loss.item()
                preds = (out >= 0.5).long()
                total_correct += (preds == labels.long()).sum().item()
                total_samples += labels.size(0)
        

        ## save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "./saved_models/best_model.pth")
            print("Best model saved.")
        
        avg_valid_loss = valid_loss / len(valid_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"Validation Loss: {avg_valid_loss:.4f}, Accuracy: {accuracy:.4f}")



# def train(model, train_loader, valid_loader, optimizer, device, loss_fn, num_epochs):
   
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

#         for batch_idx, batch in enumerate(pbar):
#             # try:
#             # Clear gradients first
#             optimizer.zero_grad()
#             graphs, labels = batch
#             graphs = graphs.to(device)
#             labels = labels.float().to(device)
#             out = model(graphs)
#             loss = loss_fn(out, labels)
#             loss.backward()


            
#             # Ensure clean computation graph for each batch
#             # with torch.enable_grad():
#             #     batch = batch.to(device)
                
#             #     # Forward pass with fresh computation graph
#             #     out = model(batch)
#             #     target = batch.y.detach().float().to(device)
                
#             #     # Detach and check for issues
#             #     if torch.isnan(out).any() or torch.isinf(out).any():
#             #         print(f"Invalid output detected in batch {batch_idx}")
#             #         continue
                
#             #     # Compute loss
#             #     loss = loss_fn(out, target)
                
#             #     if torch.isnan(loss) or torch.isinf(loss):
#             #         print(f"Invalid loss detected in batch {batch_idx}")
#             #         continue
            
#             # Backward pass
#             # loss.backward()
            
#             # Gradient clipping
#             clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             # Update parameters
#             optimizer.step()
            
#             # Track loss
#             total_loss += loss.item()
#             pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
#             # Clear variables to prevent memory leaks
#             del out, target, loss
            
#             # Periodic cleanup
#             if batch_idx % 10 == 0:
#                 torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
#             # except RuntimeError as e:
#             #     # if "backward through the graph" in str(e):
#             #     #     print(f"Graph reuse error in batch {batch_idx}. Clearing and continuing...")
#             #     #     optimizer.zero_grad()
#             #     #     # Force cleanup
#             #     #     torch.cuda.empty_cache() if torch.cuda.is_available() else None
#             #     #     continue
#             #     # elif "out of memory" in str(e):
#             #     #     print(f"OOM in batch {batch_idx}. Clearing cache...")
#             #     #     torch.cuda.empty_cache() if torch.cuda.is_available() else None
#             #     #     optimizer.zero_grad()
#             #     #     continue
#             #     # else:
#             #     #     print(f"Runtime error in batch {batch_idx}: {e}")
#             #     #     continue
#             #     breakpoint()
#             # except Exception as e:
#             #     print(f"Unexpected error in batch {batch_idx}: {e}")
#             #     continue
        
        
#         avg_loss = total_loss / len(train_loader)
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
#         model.eval()
#         total_correct = 0
#         total_samples = 0
#         total_loss = 0
#         with torch.no_grad():
#             for batch in tqdm(valid_loader, desc="Validation"):
#                 batch = batch.to(device)
#                 out = model(batch)
#                 preds = (out >= 0.5).long()
#                 loss = loss_fn(out, batch.y.float().to(device))
#                 total_loss += loss.item()
#                 total_correct += (preds == batch.y.long().to(device)).sum().item()
#                 total_samples += batch.y.size(0)

#         avg_loss = total_loss / len(valid_loader)
#         print(f"Validation Loss: {avg_loss:.4f}")
#         accuracy = total_correct / total_samples if total_samples > 0 else 0
#         print(f"Validation Accuracy: {accuracy:.4f}")



def runner():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = "icews14"
    train_path_file = f"./preds/{dataset}/train_edge_paths.pkl"
    valid_path_file = f"./preds/{dataset}/valid_edge_paths.pkl"

    rel2id = json.load(open(f"./data/{dataset}/relation2id.json", "r"))
    ts2id = json.load(open(f"./data/{dataset}/ts2id.json", "r"))

    num_rels = len(rel2id)*2 # for inverse relations
    num_ts = len(ts2id)

    ## model args - reduced batch size to avoid memory issues
    batch_size = 32  # Reduced from 32 to help with memory and graph issues
    max_len = 3
    feat_dim = 8
    hidden_dim = 128
    dropout = 0.5
    lr = 0.001

    print("Loading dataset...")
    train_data = PathGraphDataset(train_path_file, data_type="train")
    valid_data = PathGraphDataset(valid_path_file, data_type="valid")
    print(f"Dataset loaded: {len(train_data)} samples")
    print(f"Validation dataset loaded: {len(valid_data)} samples")

    # dataloader with debugging
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=train_data.collate_fn,
    )

    valid_loader = DataLoader(
        valid_data, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=valid_data.collate_fn,
    )

    print("Initializing model...")
    model = LogicalGNN(
        feat_dim=feat_dim, 
        hidden_dim=hidden_dim, 
        num_rel=num_rels, 
        num_ts=num_ts, 
        max_entities=max(train_data.max_entity, valid_data.max_entity),
        dropout=dropout, 
        max_len=max_len, 
        device=device
    )

    model = model.to(device)
    
    # Use a more conservative optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = torch.nn.BCELoss()
    num_epochs = 5
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training...")
    
    # Test with a single batch first to catch issues early
    # try:
    #     test_batch = next(iter(train_loader))
    #     test_batch = test_batch.to(device)
    #     with torch.no_grad():
    #         test_out = model(test_batch)
    #     print(f"Test forward pass successful. Output shape: {test_out.shape}")
    #     del test_batch, test_out
    #     torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # except Exception as e:
    #     print(f"Test forward pass failed: {e}")
    #     return

    train(model, train_loader, valid_loader, optimizer, device, loss_fn, num_epochs)





if __name__ == "__main__":
    runner()
    
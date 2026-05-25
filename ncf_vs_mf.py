"""
Neural Collaborative Filtering (NeuMF) vs. Matrix Factorization (SVD)
on MovieLens 100K dataset.

Metrics: RMSE (rating prediction) & Hit Rate@10 (top-K recommendation).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from surprise import SVD, Dataset as SurpriseDataset, Reader
from surprise.model_selection import train_test_split as surprise_split
import time
import requests
import zipfile
import os
from collections import defaultdict

# ------------------------------
# 1. Load MovieLens 100K (auto-download)
# ------------------------------
def load_movielens_100k():
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    if not os.path.exists("ml-100k.zip"):
        print("Downloading MovieLens 100K from", url)
        r = requests.get(url)
        with open("ml-100k.zip", "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile("ml-100k.zip", "r") as z:
            z.extractall(".")
    
    ratings = pd.read_csv("ml-100k/u.data", sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    # Convert to 0-indexed for embeddings
    ratings['user_id'] = ratings['user_id'] - 1
    ratings['item_id'] = ratings['item_id'] - 1
    
    num_users = ratings['user_id'].nunique()
    num_items = ratings['item_id'].nunique()
    
    # Chronological split (80% train, 20% test)
    ratings = ratings.sort_values('timestamp')
    split_idx = int(0.8 * len(ratings))
    train_df = ratings.iloc[:split_idx]
    test_df = ratings.iloc[split_idx:]
    
    return train_df, test_df, num_users, num_items

# ------------------------------
# 2. PyTorch Dataset for NCF
# ------------------------------
class RatingDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(df['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

# ------------------------------
# 3. NeuMF Model (NCF)
# ------------------------------
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=32, mlp_layers=[64,32,16], dropout=0.2):
        super(NeuMF, self).__init__()
        # GMF embeddings
        self.user_gmf = nn.Embedding(num_users, embed_dim)
        self.item_gmf = nn.Embedding(num_items, embed_dim)
        # MLP embeddings
        self.user_mlp = nn.Embedding(num_users, embed_dim)
        self.item_mlp = nn.Embedding(num_items, embed_dim)
        
        # MLP layers
        mlp_input = embed_dim * 2
        layers = []
        for h in mlp_layers:
            layers.append(nn.Linear(mlp_input, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            mlp_input = h
        self.mlp = nn.Sequential(*layers)
        
        # Final prediction layer (GMF output + MLP output)
        self.pred = nn.Linear(embed_dim + mlp_layers[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, user, item):
        # GMF path: element-wise product
        gmf_out = self.user_gmf(user) * self.item_gmf(item)
        # MLP path: concatenation then dense layers
        mlp_in = torch.cat([self.user_mlp(user), self.item_mlp(item)], dim=1)
        mlp_out = self.mlp(mlp_in)
        # Combine
        concat = torch.cat([gmf_out, mlp_out], dim=1)
        pred = self.pred(concat)
        # Scale to [1,5]
        pred = torch.sigmoid(pred) * 4.0 + 1.0
        return pred.squeeze()

# ------------------------------
# 4. Training function for NCF
# ------------------------------
def train_ncf(model, train_loader, val_loader, epochs=30, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_rmse = float('inf')
    patience, trigger = 5, 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for users, items, ratings in train_loader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for users, items, ratings in val_loader:
                users, items = users.to(device), items.to(device)
                preds = model(users, items)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(ratings.numpy())
        val_rmse = np.sqrt(np.mean((np.array(val_true) - np.array(val_preds))**2))
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader):.4f} | Val RMSE: {val_rmse:.4f}")
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return best_rmse

# ------------------------------
# 5. Hit Rate@K evaluation for NCF
# ------------------------------
def hit_rate_ncf(model, test_df, train_df, num_items, k=10, device='cpu'):
    model.eval()
    user_train = defaultdict(set)
    for _, row in train_df.iterrows():
        user_train[row['user_id']].add(row['item_id'])
    
    user_test = defaultdict(list)
    for _, row in test_df.iterrows():
        user_test[row['user_id']].append(row['item_id'])
    
    hits, total = 0, 0
    with torch.no_grad():
        for user, test_items in user_test.items():
            if not test_items:
                continue
            total += 1
            seen = user_train[user]
            candidates = [i for i in range(num_items) if i not in seen]
            if not candidates:
                continue
            user_t = torch.tensor([user] * len(candidates), dtype=torch.long).to(device)
            item_t = torch.tensor(candidates, dtype=torch.long).to(device)
            scores = model(user_t, item_t).cpu().numpy()
            top_k = [candidates[i] for i in np.argsort(scores)[-k:][::-1]]
            if any(item in top_k for item in test_items):
                hits += 1
    return hits / total if total > 0 else 0

# ------------------------------
# 6. Hit Rate@K for SVD (baseline)
# ------------------------------
def hit_rate_svd(svd_model, test_df, train_df, num_items, k=10):
    user_train = defaultdict(set)
    for _, row in train_df.iterrows():
        user_train[row['user_id']].add(row['item_id'])
    user_test = defaultdict(list)
    for _, row in test_df.iterrows():
        user_test[row['user_id']].append(row['item_id'])
    
    hits, total = 0, 0
    for user, test_items in user_test.items():
        if not test_items:
            continue
        total += 1
        seen = user_train[user]
        candidates = [i for i in range(num_items) if i not in seen]
        if not candidates:
            continue
        preds = [(i, svd_model.predict(user, i).est) for i in candidates]
        preds.sort(key=lambda x: x[1], reverse=True)
        top_k = [i for i, _ in preds[:k]]
        if any(item in top_k for item in test_items):
            hits += 1
    return hits / total if total > 0 else 0

# ------------------------------
# 7. Main experiment
# ------------------------------
def main():
    print("Loading MovieLens 100K...")
    train_df, test_df, num_users, num_items = load_movielens_100k()
    print(f"Users: {num_users}, Items: {num_items}, Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Prepare NCF data loaders
    dataset = RatingDataset(train_df)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_sub, val_sub = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_sub, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=256, shuffle=False)
    test_loader = DataLoader(RatingDataset(test_df), batch_size=256, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ---------- Baseline: SVD (Matrix Factorization) ----------
    print("\n--- Training Matrix Factorization (SVD) ---")
    reader = Reader(rating_scale=(1,5))
    surprise_data = SurpriseDataset.load_from_df(train_df[['user_id','item_id','rating']], reader)
    trainset, valset = surprise_split(surprise_data, test_size=0.2, random_state=42)
    svd = SVD(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02)
    start = time.time()
    svd.fit(trainset)
    svd_time = time.time() - start
    
    # SVD test RMSE
    testset = list(zip(test_df['user_id'], test_df['item_id'], test_df['rating']))
    svd_preds = [svd.predict(uid, iid).est for (uid, iid, _) in testset]
    svd_rmse = np.sqrt(np.mean((test_df['rating'].values - np.array(svd_preds))**2))
    svd_hit = hit_rate_svd(svd, test_df, train_df, num_items, k=10)
    print(f"SVD RMSE: {svd_rmse:.4f}, Hit Rate@10: {svd_hit:.4f}, Time: {svd_time:.2f}s")
    
    # ---------- NCF (NeuMF) ----------
    print("\n--- Training Neural Collaborative Filtering (NeuMF) ---")
    ncf_model = NeuMF(num_users, num_items, embed_dim=32, mlp_layers=[64,32,16])
    start = time.time()
    train_ncf(ncf_model, train_loader, val_loader, epochs=30, lr=0.001, device=device)
    ncf_time = time.time() - start
    
    # NCF test evaluation
    ncf_model.eval()
    ncf_preds, ncf_true = [], []
    with torch.no_grad():
        for users, items, ratings in test_loader:
            users, items = users.to(device), items.to(device)
            preds = ncf_model(users, items)
            ncf_preds.extend(preds.cpu().numpy())
            ncf_true.extend(ratings.numpy())
    ncf_rmse = np.sqrt(np.mean((np.array(ncf_true) - np.array(ncf_preds))**2))
    ncf_hit = hit_rate_ncf(ncf_model, test_df, train_df, num_items, k=10, device=device)
    print(f"NCF RMSE: {ncf_rmse:.4f}, Hit Rate@10: {ncf_hit:.4f}, Time: {ncf_time:.2f}s")
    
    # ---------- Final Comparison ----------
    print("\n" + "="*60)
    print("Final Comparison")
    print("="*60)
    print(f"{'Model':<30} {'RMSE':<10} {'Hit Rate@10':<12} {'Time (s)':<10}")
    print("-"*60)
    print(f"{'Matrix Factorization (SVD)':<30} {svd_rmse:<10.4f} {svd_hit:<12.4f} {svd_time:<10.2f}")
    print(f"{'Neural Collaborative Filtering':<30} {ncf_rmse:<10.4f} {ncf_hit:<12.4f} {ncf_time:<10.2f}")
    print("="*60)

if __name__ == "__main__":
    main()
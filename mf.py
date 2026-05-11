"""
MF — Matrix Factorization
=========================
Pure user-item matrix factorization, no side features.

  y_hat = global_bias + u_bias + i_bias + <u_emb, i_emb>

A baseline for comparison against the side-feature-augmented model (fm_demo.py).

Dataset: MovieLens 100k — 943 users, 1682 items, 100k ratings.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

# reuse data loader from fm_demo
from fm_demo import download_data, load_ratings

N_USERS = 943
N_ITEMS = 1682


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
class MFDataset(Dataset):
    def __init__(self, u, i, y, indices):
        self.u = torch.tensor(u[indices], dtype=torch.long)
        self.i = torch.tensor(i[indices], dtype=torch.long)
        self.y = torch.tensor(y[indices], dtype=torch.float32)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return {
            "user":  self.u[idx],
            "item":  self.i[idx],
            "label": self.y[idx],
        }


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
class MF(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.dim = dim

        self.u_emb = nn.Embedding(n_users, dim)
        self.i_emb = nn.Embedding(n_items, dim)
        self.u_bias = nn.Embedding(n_users, 1)
        self.i_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.u_emb.weight, std=0.01)
        nn.init.normal_(self.i_emb.weight, std=0.01)
        nn.init.zeros_(self.u_bias.weight)
        nn.init.zeros_(self.i_bias.weight)

    def forward(self, batch):
        ue = self.u_emb(batch["user"])
        ie = self.i_emb(batch["item"])
        pair = (ue * ie).sum(dim=1)
        bias = (self.u_bias(batch["user"]).squeeze(-1)
                + self.i_bias(batch["item"]).squeeze(-1)
                + self.global_bias)
        return pair + bias

    @torch.no_grad()
    def build_user_vectors(self, user_ids):
        """user_vec = [u_emb, u_bias, 1]  → dim + 2"""
        return torch.cat([
            self.u_emb(user_ids),
            self.u_bias(user_ids),
            torch.ones(len(user_ids), 1),
        ], dim=1)

    @torch.no_grad()
    def build_item_vectors(self, item_ids):
        """item_vec = [i_emb, 1, i_bias + global_bias]  → dim + 2"""
        n = len(item_ids)
        return torch.cat([
            self.i_emb(item_ids),
            torch.ones(n, 1),
            self.i_bias(item_ids) + self.global_bias,
        ], dim=1)

    @torch.no_grad()
    def ctr(self, user_vecs, item_vecs):
        return torch.sigmoid((user_vecs * item_vecs).sum(dim=1))


def run_epoch(model, loader, optim=None):
    is_train = optim is not None
    model.train(is_train)
    total_loss = 0
    logits_all, labels_all = [], []
    for batch in loader:
        logits = model(batch)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, batch["label"])
        if is_train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        total_loss += loss.item()
        logits_all.append(logits.detach())
        labels_all.append(batch["label"])
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    loss = total_loss / len(loader)
    auc = roc_auc_score(labels.numpy(), torch.sigmoid(logits).numpy())
    return loss, auc


def main():
    print("=" * 50)
    print("  MF — Matrix Factorization")
    print("=" * 50)

    u, i, y = load_ratings()
    n = len(y)
    perm = np.random.RandomState(42).permutation(n)
    train_idx = perm[:int(0.8 * n)]
    val_idx   = perm[int(0.8 * n):int(0.9 * n)]
    test_idx  = perm[int(0.9 * n):]

    train_ds = MFDataset(u, i, y, train_idx)
    val_ds   = MFDataset(u, i, y, val_idx)
    test_ds  = MFDataset(u, i, y, test_idx)

    train_loader = DataLoader(train_ds, 1024, shuffle=True)
    val_loader   = DataLoader(val_ds, 1024)
    test_loader  = DataLoader(test_ds, 1024)

    model = MF(N_USERS, N_ITEMS, dim=64)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 5, 0.5)

    print(f"\n  {N_USERS} users, {N_ITEMS} items, {n} ratings,"
          f" emb_dim={model.dim}")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Training...\n")

    best_auc = 0
    for epoch in range(20):
        tr_loss, tr_auc = run_epoch(model, train_loader, optim)
        val_loss, val_auc = run_epoch(model, val_loader)
        scheduler.step()
        mark = "  ★" if val_auc > best_auc else ""
        if val_auc > best_auc:
            best_auc = val_auc
        print(f"  E{epoch+1:2d}  train_loss={tr_loss:.4f} auc={tr_auc:.4f}"
              f"  val_loss={val_loss:.4f} auc={val_auc:.4f}{mark}")

    test_loss, test_auc = run_epoch(model, test_loader)
    print(f"\n  Test  loss={test_loss:.4f}  AUC={test_auc:.4f}")

    print("\n  Building serving vectors ...")
    user_vecs = model.build_user_vectors(torch.arange(N_USERS))
    item_vecs = model.build_item_vectors(torch.arange(N_ITEMS))
    print(f"  user_vec: {user_vecs.shape}  item_vec: {item_vecs.shape}")

    # compare to model output on test set
    with torch.no_grad():
        vec_ctr = model.ctr(user_vecs[u[test_idx]], item_vecs[i[test_idx]])
    vec_auc = roc_auc_score(y[test_idx], vec_ctr.numpy())
    match = "✓" if abs(vec_auc - test_auc) < 0.005 else "△"
    print(f"  Vector AUC={vec_auc:.4f}  (model AUC={test_auc:.4f})  {match}")

    demo_uid = 42
    with torch.no_grad():
        ctrs = model.ctr(user_vecs[demo_uid:demo_uid + 1], item_vecs).squeeze(0)
        top_vals, top_ids = ctrs.topk(10)

    from fm_demo import load_meta
    _, titles, _ = load_meta()
    print(f"\n  User {demo_uid+1} — Top-10 (MF):")
    for rk, idx in enumerate(top_ids.tolist()):
        print(f"    {rk+1}. [{idx+1}] {titles.get(idx, '?')}"
              f"  ctr={ctrs[idx]:.4f}")


if __name__ == "__main__":
    main()

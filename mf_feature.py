"""
MF — No user ID, feature-based only
=====================================
User is represented purely by occupation feature (one-hot → projection),
no user_id embedding. Demonstrates content-based recommendation.

Item side still uses item_id + genre for reasonable AUC.

This simulates a cold-start scenario where new users have no history,
only profile features.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from fm_demo import download_data, load_ratings, load_meta

N_USERS = 943
N_ITEMS = 1682
N_OCC = 21


def load_occupation_features():
    """Return [n_users, N_OCC] one-hot occupation matrix."""
    download_data()
    occ_names = [l.strip() for l in open("ml-100k/u.occupation")]
    occ2id = {n: i for i, n in enumerate(occ_names)}
    occs = np.zeros((N_USERS, N_OCC), dtype=np.float32)
    with open("ml-100k/u.user") as f:
        for line in f:
            p = line.strip().split("|")
            uid = int(p[0]) - 1
            occs[uid, occ2id[p[3]]] = 1.0
    return occs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class FeatDataset(Dataset):
    def __init__(self, u, i, y, occ_feat, item_genres, indices):
        self.u = u[indices]
        self.i = i[indices]
        self.y = y[indices]
        self.occ = torch.tensor(occ_feat[self.u], dtype=torch.float32)
        self.iid = torch.tensor(self.i, dtype=torch.long)
        self.labels = torch.tensor(self.y, dtype=torch.float32)
        self.genres = [item_genres.get(iid, []) for iid in self.i]

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        g = np.zeros(19, dtype=np.float32)
        g[self.genres[idx]] = 1.0
        return {
            "occ":    self.occ[idx],
            "item":   self.iid[idx],
            "genres": torch.tensor(g),
            "label":  self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Model — no user_id embedding
# ---------------------------------------------------------------------------
class FeatMF(nn.Module):
    def __init__(self, n_items, n_occ=21, n_genres=19, dim=64):
        super().__init__()
        self.dim = dim
        # User side: occupation one-hot → linear projection (no ID embedding)
        self.u_proj = nn.Linear(n_occ, dim, bias=False)
        # Item side
        self.i_emb = nn.Embedding(n_items, dim)
        self.g_emb = nn.Embedding(n_genres, dim)
        # Biases
        self.u_bias = nn.Linear(n_occ, 1, bias=False)
        self.i_bias = nn.Embedding(n_items, 1)
        self.g_bias = nn.Parameter(torch.zeros(1))
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.u_proj.weight, std=0.01)
        nn.init.normal_(self.i_emb.weight, std=0.01)
        nn.init.normal_(self.g_emb.weight, std=0.01)
        nn.init.zeros_(self.u_bias.weight)
        nn.init.zeros_(self.i_bias.weight)

    def genre_agg(self, genres):
        mask = genres.unsqueeze(-1)
        g = self.g_emb.weight.unsqueeze(0) * mask
        return g.sum(dim=1)

    def forward(self, batch):
        # User: feature → embedding (no ID lookup)
        ue = self.u_proj(batch["occ"])
        ub = self.u_bias(batch["occ"]).squeeze(-1)
        # Item
        ie = self.i_emb(batch["item"]) + self.genre_agg(batch["genres"])
        ib = self.i_bias(batch["item"]).squeeze(-1)
        pair = (ue * ie).sum(dim=1)
        bias = ub + ib + self.g_bias + self.global_bias
        return pair + bias

    @torch.no_grad()
    def build_user_vectors(self, occ_feat):
        """user_vec = [u_proj(occ), u_bias(occ), 1, 1]  → dim + 3"""
        ue = self.u_proj(occ_feat)
        ub = self.u_bias(occ_feat)
        n = len(occ_feat)
        return torch.cat([ue, ub, torch.ones(n, 1), torch.ones(n, 1), torch.ones(n, 1)], dim=1)

    @torch.no_grad()
    def build_item_vectors(self, item_ids, genre_lists):
        """item_vec = [i_emb + g_agg, 1, 1, i_bias + g_bias + global_bias]"""
        ie = self.i_emb(item_ids)
        d = self.dim
        n = len(item_ids)
        g_agg = torch.zeros(n, d)
        for idx, gids in enumerate(genre_lists):
            if gids:
                g_agg[idx] = self.g_emb.weight[gids].sum(dim=0)
        ib = self.i_bias(item_ids)
        gb = (self.g_bias + self.global_bias).expand(n, 1)
        return torch.cat([ie + g_agg, torch.ones(n, 1), torch.ones(n, 1), ib, gb], dim=1)

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
    print("=" * 60)
    print("  FeatMF — No user ID, occupation features only")
    print("=" * 60)

    u, i, y = load_ratings()
    _, _, item_genres = load_meta()
    occ_feat = torch.tensor(load_occupation_features(), dtype=torch.float32)

    n = len(y)
    perm = np.random.RandomState(42).permutation(n)
    train_idx = perm[:int(0.8 * n)]
    val_idx   = perm[int(0.8 * n):int(0.9 * n)]
    test_idx  = perm[int(0.9 * n):]

    train_ds = FeatDataset(u, i, y, occ_feat.numpy(), item_genres, train_idx)
    val_ds   = FeatDataset(u, i, y, occ_feat.numpy(), item_genres, val_idx)
    test_ds  = FeatDataset(u, i, y, occ_feat.numpy(), item_genres, test_idx)

    train_loader = DataLoader(train_ds, 1024, shuffle=True)
    val_loader   = DataLoader(val_ds, 1024)
    test_loader  = DataLoader(test_ds, 1024)

    model = FeatMF(N_ITEMS, dim=64)
    print(f"\n  params: {sum(p.numel() for p in model.parameters()):,}")
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    best_auc = 0
    for epoch in range(20):
        tr_loss, tr_auc = run_epoch(model, train_loader, optim)
        val_loss, val_auc = run_epoch(model, val_loader)
        mark = "  ★" if val_auc > best_auc else ""
        if val_auc > best_auc:
            best_auc = val_auc
        print(f"  E{epoch+1:2d}  train_loss={tr_loss:.4f} auc={tr_auc:.4f}"
              f"  val_loss={val_loss:.4f} auc={val_auc:.4f}{mark}")

    test_loss, test_auc = run_epoch(model, test_loader)
    print(f"\n  Test  loss={test_loss:.4f}  AUC={test_auc:.4f}")

    # Build vectors
    user_vecs = model.build_user_vectors(occ_feat)
    item_vecs = model.build_item_vectors(
        torch.arange(N_ITEMS),
        [item_genres.get(i, []) for i in range(N_ITEMS)])
    print(f"  user_vec: {user_vecs.shape}  item_vec: {item_vecs.shape}")

    with torch.no_grad():
        vec_ctr = model.ctr(user_vecs[u[test_idx]], item_vecs[i[test_idx]])
    vec_auc = roc_auc_score(y[test_idx], vec_ctr.numpy())
    match = "✓" if abs(vec_auc - test_auc) < 0.005 else "△"
    print(f"  Vector AUC={vec_auc:.4f}  (model AUC={test_auc:.4f})  {match}")

    # Compare with pure MF baseline
    print(f"\n  Comparison (test AUC):")
    print(f"    MF (user_id)          ~0.74")
    print(f"    FeatMF (occupation)   {test_auc:.4f}")


if __name__ == "__main__":
    main()

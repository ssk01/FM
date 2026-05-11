"""
ablation.py — 消融实验：量化 occ / genre 各自贡献
=================================================
训练 4 个模型，仅 user/item 侧特征不同，其余完全一致。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from fm_demo import download_data, load_ratings, load_meta

N_USERS, N_ITEMS, N_OCC, N_GENRES = 943, 1682, 21, 19
DIM = 64


# ---------------------------------------------------------------------------
# Dataset (通用)
# ---------------------------------------------------------------------------
class BaseDataset(Dataset):
    def __init__(self, u, i, y, occs, item_genres, indices):
        u, i = u[indices], i[indices]
        self.user = torch.tensor(u, dtype=torch.long)
        self.item = torch.tensor(i, dtype=torch.long)
        self.occ = torch.tensor([occs.get(uid, 0) for uid in u], dtype=torch.long)
        g = np.zeros((len(u), N_GENRES), dtype=np.float32)
        for idx, iid in enumerate(i):
            gidx = item_genres.get(iid, [])
            g[idx, gidx] = 1.0
        self.genres = torch.tensor(g)
        self.label = torch.tensor(y[indices], dtype=torch.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            "user": self.user[idx], "item": self.item[idx],
            "occ": self.occ[idx], "genres": self.genres[idx],
            "label": self.label[idx],
        }


# ---------------------------------------------------------------------------
# 4 种模型配置
# ---------------------------------------------------------------------------
def make_mf():
    """user_id, item_id"""
    class MF(nn.Module):
        def __init__(self):
            super().__init__()
            self.u = nn.Embedding(N_USERS, DIM)
            self.i = nn.Embedding(N_ITEMS, DIM)
            self.bu = nn.Embedding(N_USERS, 1)
            self.bi = nn.Embedding(N_ITEMS, 1)
            self.b0 = nn.Parameter(torch.zeros(1))
            self._init()

        def _init(self):
            for p in [self.u, self.i]:
                nn.init.normal_(p.weight, std=0.01)
            for p in [self.bu, self.bi]:
                nn.init.zeros_(p.weight)

        def forward(self, b):
            logit = (self.u(b["user"]) * self.i(b["item"])).sum(dim=1)
            logit += self.bu(b["user"]).squeeze() + self.bi(b["item"]).squeeze() + self.b0
            return logit
    return MF()


def make_mf_occ():
    """user_id + occ, item_id"""
    class MFOcc(nn.Module):
        def __init__(self):
            super().__init__()
            self.u = nn.Embedding(N_USERS, DIM)
            self.o = nn.Embedding(N_OCC, DIM)
            self.i = nn.Embedding(N_ITEMS, DIM)
            self.bu = nn.Embedding(N_USERS, 1)
            self.bo = nn.Embedding(N_OCC, 1)
            self.bi = nn.Embedding(N_ITEMS, 1)
            self.b0 = nn.Parameter(torch.zeros(1))
            self._init()

        def _init(self):
            for p in [self.u, self.o, self.i]:
                nn.init.normal_(p.weight, std=0.01)
            for p in [self.bu, self.bo, self.bi]:
                nn.init.zeros_(p.weight)

        def forward(self, b):
            ue = self.u(b["user"]) + self.o(b["occ"])
            logit = (ue * self.i(b["item"])).sum(dim=1)
            logit += (self.bu(b["user"]) + self.bo(b["occ"])).squeeze()
            logit += self.bi(b["item"]).squeeze() + self.b0
            return logit
    return MFOcc()


def make_mf_genre():
    """user_id, item_id + genre"""
    class MFGenre(nn.Module):
        def __init__(self):
            super().__init__()
            self.u = nn.Embedding(N_USERS, DIM)
            self.i = nn.Embedding(N_ITEMS, DIM)
            self.g = nn.Embedding(N_GENRES, DIM)
            self.bu = nn.Embedding(N_USERS, 1)
            self.bi = nn.Embedding(N_ITEMS, 1)
            self.bg = nn.Parameter(torch.zeros(1))
            self.b0 = nn.Parameter(torch.zeros(1))
            self._init()

        def _init(self):
            for p in [self.u, self.i, self.g]:
                nn.init.normal_(p.weight, std=0.01)
            for p in [self.bu, self.bi]:
                nn.init.zeros_(p.weight)

        def forward(self, b):
            mask = b["genres"].unsqueeze(-1)
            g_agg = (self.g.weight.unsqueeze(0) * mask).sum(dim=1)
            ie = self.i(b["item"]) + g_agg
            logit = (self.u(b["user"]) * ie).sum(dim=1)
            logit += self.bu(b["user"]).squeeze() + self.bi(b["item"]).squeeze()
            logit += self.bg + self.b0
            return logit
    return MFGenre()


def make_full():
    """user_id + occ, item_id + genre"""
    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.u = nn.Embedding(N_USERS, DIM)
            self.o = nn.Embedding(N_OCC, DIM)
            self.i = nn.Embedding(N_ITEMS, DIM)
            self.g = nn.Embedding(N_GENRES, DIM)
            self.bu = nn.Embedding(N_USERS, 1)
            self.bo = nn.Embedding(N_OCC, 1)
            self.bi = nn.Embedding(N_ITEMS, 1)
            self.bg = nn.Parameter(torch.zeros(1))
            self.b0 = nn.Parameter(torch.zeros(1))
            self._init()

        def _init(self):
            for p in [self.u, self.o, self.i, self.g]:
                nn.init.normal_(p.weight, std=0.01)
            for p in [self.bu, self.bo, self.bi]:
                nn.init.zeros_(p.weight)

        def forward(self, b):
            ue = self.u(b["user"]) + self.o(b["occ"])
            mask = b["genres"].unsqueeze(-1)
            g_agg = (self.g.weight.unsqueeze(0) * mask).sum(dim=1)
            ie = self.i(b["item"]) + g_agg
            logit = (ue * ie).sum(dim=1)
            logit += (self.bu(b["user"]) + self.bo(b["occ"])).squeeze()
            logit += self.bi(b["item"]).squeeze() + self.bg + self.b0
            return logit
    return Full()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optim=None):
    is_train = optim is not None
    model.train(is_train)
    total_loss = 0
    logits_all, labels_all = [], []
    for b in loader:
        logits = model(b)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, b["label"])
        if is_train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        total_loss += loss.item()
        logits_all.append(logits.detach())
        labels_all.append(b["label"])
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    loss = total_loss / len(loader)
    auc = roc_auc_score(labels.numpy(), torch.sigmoid(logits).numpy())
    return loss, auc


def train_model(model, train_loader, val_loader, name):
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    best = -1
    best_state = None
    for epoch in range(20):
        tr_l, tr_a = run_epoch(model, train_loader, optim)
        va_l, va_a = run_epoch(model, val_loader)
        mark = ""
        if va_a > best:
            best = va_a
            best_state = model.state_dict()
            mark = "  ★"
        if epoch < 3 or epoch == 19:
            print(f"  {name:10s} E{epoch+1:2d}  train_loss={tr_l:.4f} auc={tr_a:.4f}"
                  f"  val_loss={va_l:.4f} auc={va_a:.4f}{mark}")
    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 58)
    print("  Ablation — Quantifying occ / genre contribution")
    print("=" * 58)

    u, i, y = load_ratings()
    occs, _, item_genres = load_meta()

    np.random.seed(42)
    perm = np.random.permutation(len(y))
    train_idx = perm[:int(0.8 * len(y))]
    val_idx   = perm[int(0.8 * len(y)):int(0.9 * len(y))]
    test_idx  = perm[int(0.9 * len(y)):]

    train_ds = BaseDataset(u, i, y, occs, item_genres, train_idx)
    val_ds   = BaseDataset(u, i, y, occs, item_genres, val_idx)
    test_ds  = BaseDataset(u, i, y, occs, item_genres, test_idx)

    train_lo = DataLoader(train_ds, 1024, shuffle=True)
    val_lo   = DataLoader(val_ds, 1024)
    test_lo  = DataLoader(test_ds, 1024)

    configs = [
        ("MF",        make_mf),
        ("MF+occ",    make_mf_occ),
        ("MF+genre",  make_mf_genre),
        ("MF+occ+genre", make_full),
    ]

    results = {}
    for name, maker in configs:
        model = maker()
        params = sum(p.numel() for p in model.parameters())
        print(f"\n  ── {name} ({params:,} params) ──")
        train_model(model, train_lo, val_lo, name)
        test_l, test_a = run_epoch(model, test_lo)
        results[name] = f"{test_a:.4f}"
        print(f"  {name:10s} TEST auc={test_a:.4f}")

    print("\n" + "=" * 58)
    print("  Summary")
    print("=" * 58)
    for name, auc_str in results.items():
        print(f"  {name:15s}  Test AUC = {auc_str}")
    print(f"\n  Δ(occ)   = {name} - MF      = MF+occ AUC - MF AUC")
    print(f"  Δ(genre) = MF+genre - MF")
    print(f"  Δ(full)  = MF+occ+genre - MF")


if __name__ == "__main__":
    main()

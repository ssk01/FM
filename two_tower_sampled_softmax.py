"""
two_tower_sampled_softmax.py — 双塔 + in-batch sampled softmax
================================================================
和 two_tower_norm.py 同样的归一化双塔，但训练目标换成 in-batch sampled softmax
（即 InfoNCE / SimCLR / 双塔召回常用 loss）：

  对每个 batch 取 B 条正样本 (u_k, i_k)，构造 BxB logit 矩阵：
      L[a, b] = <u_a, i_b> / τ
  对角线为正样本，每行其它 B-1 列作为 in-batch 负样本。
  loss = mean( CrossEntropy(row_a, target=a) )

特点：
  - 训练只用正样本（rating >= 4），负样本由 batch 内其它 item 提供（无标签即可训练）
  - τ = 0.07
  - 评估时仍用全量 test 集（含正负），算 AUC（与 norm+BCE 版本可比）
  - 没有做 logQ correction（按热门度修正采样偏差）；如要做，可在 logits 上减去 log p(item)

注意 false negative：batch 内可能恰好出现同一个 item 或对该 user 也是正的 item，
会被当成负样本压低 —— 实践上接受为噪声，或加 mask；这里保持最简实现。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

from fm_demo import prepare_data
from two_tower_norm import TwoTowerNorm

TEMPERATURE = 0.07
BATCH = 1024


# ---------------------------------------------------------------------------
# Dataset：只产出正样本（rating >= 4）
# ---------------------------------------------------------------------------
class PositiveDataset(Dataset):
    def __init__(self, data, indices):
        mask = data["y"][indices] > 0.5
        self.idx = indices[mask]
        self.u = data["u"]
        self.i = data["i"]
        self.occs = data["occs"]
        self.item_genres = data["item_genres"]
        self.n_genres = data["n_genres"]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        idx = self.idx[k]
        uid = int(self.u[idx])
        iid = int(self.i[idx])
        occ = self.occs.get(uid, 0)
        g = np.zeros(self.n_genres, dtype=np.float32)
        for gid in self.item_genres.get(iid, []):
            g[gid] = 1.0
        return {
            "user":   torch.tensor(uid, dtype=torch.long),
            "item":   torch.tensor(iid, dtype=torch.long),
            "occ":    torch.tensor(occ, dtype=torch.long),
            "genres": torch.tensor(g),
        }


# ---------------------------------------------------------------------------
# 评估 dataset：保留正负样本和 label，与前面两版可比
# ---------------------------------------------------------------------------
class EvalDataset(Dataset):
    def __init__(self, data, indices):
        self.idx = indices
        self.u = data["u"]
        self.i = data["i"]
        self.y = data["y"]
        self.occs = data["occs"]
        self.item_genres = data["item_genres"]
        self.n_genres = data["n_genres"]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        idx = self.idx[k]
        uid = int(self.u[idx])
        iid = int(self.i[idx])
        occ = self.occs.get(uid, 0)
        g = np.zeros(self.n_genres, dtype=np.float32)
        for gid in self.item_genres.get(iid, []):
            g[gid] = 1.0
        return {
            "user":   torch.tensor(uid, dtype=torch.long),
            "item":   torch.tensor(iid, dtype=torch.long),
            "occ":    torch.tensor(occ, dtype=torch.long),
            "genres": torch.tensor(g),
            "label":  torch.tensor(self.y[idx], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optim):
    model.train()
    total_loss, n_batches = 0.0, 0
    correct, total = 0, 0
    for batch in loader:
        u = model.user_tower(batch["user"], batch["occ"])    # [B, d]
        i = model.item_tower(batch["item"], batch["genres"]) # [B, d]
        logits = u @ i.t() / model.tau                       # [B, B]
        target = torch.arange(u.size(0))
        loss = F.cross_entropy(logits, target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
        n_batches += 1
        correct += (logits.argmax(-1) == target).sum().item()
        total += u.size(0)
    return total_loss / n_batches, correct / total


@torch.no_grad()
def eval_auc(model, loader):
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        logits = model(batch)   # 走 forward，cosine / τ
        logits_all.append(logits)
        labels_all.append(batch["label"])
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    return roc_auc_score(labels.numpy(), torch.sigmoid(logits).numpy())


def main():
    print("=" * 60)
    print(f"  TwoTower + Sampled Softmax (in-batch), τ = {TEMPERATURE}")
    print("=" * 60)

    data = prepare_data()
    train_ds = PositiveDataset(data, data["train"])
    val_ds   = EvalDataset(data, data["val"])
    test_ds  = EvalDataset(data, data["test"])
    print(f"  train positives: {len(train_ds)} / {len(data['train'])}")

    train_loader = DataLoader(train_ds, BATCH, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   BATCH)
    test_loader  = DataLoader(test_ds,  BATCH)

    model = TwoTowerNorm(data["n_users"], data["n_items"], dim=64,
                         n_occ=data["n_occ"], n_genres=data["n_genres"],
                         temperature=TEMPERATURE)
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(optim, 5, 0.5)

    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  batch size {BATCH} → 每条正样本配 {BATCH-1} in-batch 负\n")

    best_val = 0
    for ep in range(20):
        loss, acc = train_epoch(model, train_loader, optim)
        va_auc = eval_auc(model, val_loader)
        sched.step()
        mark = ""
        if va_auc > best_val:
            best_val = va_auc
            torch.save(model.state_dict(), "two_tower_sampled_softmax.pt")
            mark = "  ★"
        print(f"  E{ep+1:2d}  ce_loss={loss:.4f}  in-batch acc={acc:.3f}  "
              f"val_auc={va_auc:.4f}{mark}")

    model.load_state_dict(torch.load("two_tower_sampled_softmax.pt"))
    te_auc = eval_auc(model, test_loader)
    print(f"\n  Test AUC = {te_auc:.4f}")

    # 向量校验
    user_occ = torch.zeros(data["n_users"], dtype=torch.long)
    for u in range(data["n_users"]):
        user_occ[u] = data["occs"].get(u, 0)
    g_full = torch.zeros(data["n_items"], data["n_genres"])
    for i in range(data["n_items"]):
        for k in data["item_genres"].get(i, []):
            g_full[i, k] = 1.0
    user_vecs = model.build_user_vectors(torch.arange(data["n_users"]), user_occ)
    item_vecs = model.build_item_vectors(torch.arange(data["n_items"]), g_full)
    print(f"\n  user_vec={tuple(user_vecs.shape)}  item_vec={tuple(item_vecs.shape)}")
    print(f"  ||user_vec||={user_vecs.norm(dim=-1).mean():.4f}  "
          f"||item_vec||={item_vecs.norm(dim=-1).mean():.4f}")

    # Top-10 召回 demo（cosine，可直接走 FAISS IndexFlatIP）
    demo_uid = 42
    cos = (user_vecs[demo_uid:demo_uid+1] * item_vecs).sum(-1)
    top_vals, top_ids = cos.topk(10)
    titles = data["titles"]
    print(f"\n  User {demo_uid+1} — Top-10 (cosine):")
    for rk, idx in enumerate(top_ids.tolist()):
        print(f"    {rk+1}. [{idx+1}] {titles.get(idx, '?')}  cos={cos[idx]:.4f}")


if __name__ == "__main__":
    main()

"""
two_tower_norm.py — 双塔 + L2 norm + temperature
=================================================
和 fm_demo.py 相同的两侧 sum-pooling 结构，但：
  - user_vec / item_vec 都做 L2 归一化（cosine 相似度）
  - logit = cosine / τ，τ = 0.07
  - 损失：BCEWithLogits

τ=0.07 是 SimCLR / CLIP 默认值；cos 在 [-1,1]，除以 0.07 拉到 [-14.3, 14.3]，
正好匹配 sigmoid 的有效区间。

注：归一化双塔不再保留一阶 bias —— bias 加在归一化之外会破坏 cosine 几何。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from fm_demo import prepare_data, FMDataset

TEMPERATURE = 0.07


class TwoTowerNorm(nn.Module):
    def __init__(self, n_users, n_items, dim=64, n_occ=21, n_genres=19,
                 temperature=TEMPERATURE):
        super().__init__()
        self.dim = dim
        self.tau = temperature

        self.u_emb = nn.Embedding(n_users, dim)
        self.o_emb = nn.Embedding(n_occ, dim)
        self.i_emb = nn.Embedding(n_items, dim)
        self.g_emb = nn.Embedding(n_genres, dim)

        for emb in [self.u_emb, self.o_emb, self.i_emb, self.g_emb]:
            nn.init.normal_(emb.weight, std=0.01)

    def user_tower(self, user_ids, occ_ids):
        v = self.u_emb(user_ids) + self.o_emb(occ_ids)
        return F.normalize(v, dim=-1)

    def item_tower(self, item_ids, genres_multi_hot):
        ie = self.i_emb(item_ids)
        ge = (self.g_emb.weight.unsqueeze(0) * genres_multi_hot.unsqueeze(-1)).sum(1)
        return F.normalize(ie + ge, dim=-1)

    def forward(self, batch):
        u = self.user_tower(batch["user"], batch["occ"])
        i = self.item_tower(batch["item"], batch["genres"])
        return (u * i).sum(-1) / self.tau

    @torch.no_grad()
    def build_user_vectors(self, user_ids, occ_ids):
        return self.user_tower(user_ids, occ_ids)

    @torch.no_grad()
    def build_item_vectors(self, item_ids, genres_multi_hot):
        return self.item_tower(item_ids, genres_multi_hot)


def run_epoch(model, loader, optim=None):
    is_train = optim is not None
    model.train(is_train)
    total_loss = 0
    logits_all, labels_all = [], []
    for batch in loader:
        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch["label"])
        if is_train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        total_loss += loss.item()
        logits_all.append(logits.detach())
        labels_all.append(batch["label"])
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    return total_loss / len(loader), roc_auc_score(labels.numpy(),
                                                   torch.sigmoid(logits).numpy())


def main():
    print("=" * 60)
    print(f"  TwoTowerNorm — L2-normalized 双塔, τ = {TEMPERATURE}")
    print("=" * 60)

    data = prepare_data()
    train_loader = DataLoader(FMDataset(data, data["train"]), 1024, shuffle=True)
    val_loader   = DataLoader(FMDataset(data, data["val"]),   1024)
    test_loader  = DataLoader(FMDataset(data, data["test"]),  1024)

    model = TwoTowerNorm(data["n_users"], data["n_items"], dim=64,
                         n_occ=data["n_occ"], n_genres=data["n_genres"])
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(optim, 5, 0.5)

    print(f"  params: {sum(p.numel() for p in model.parameters()):,}\n")
    best_val = 0
    for ep in range(20):
        tr_loss, tr_auc = run_epoch(model, train_loader, optim)
        va_loss, va_auc = run_epoch(model, val_loader)
        sched.step()
        mark = ""
        if va_auc > best_val:
            best_val = va_auc
            torch.save(model.state_dict(), "two_tower_norm.pt")
            mark = "  ★"
        print(f"  E{ep+1:2d}  tr={tr_loss:.4f}/{tr_auc:.4f}  "
              f"va={va_loss:.4f}/{va_auc:.4f}{mark}")

    model.load_state_dict(torch.load("two_tower_norm.pt"))
    te_loss, te_auc = run_epoch(model, test_loader)
    print(f"\n  Test  loss={te_loss:.4f}  AUC={te_auc:.4f}")

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
          f"||item_vec||={item_vecs.norm(dim=-1).mean():.4f}  (应 ≈ 1.0)")

    # FAISS-friendly：直接 IP 检索就是 cosine 检索
    tu = data["u"][data["test"]]
    ti = data["i"][data["test"]]
    ty = data["y"][data["test"]]
    with torch.no_grad():
        cos = (user_vecs[tu] * item_vecs[ti]).sum(-1)
        vec_logits = cos / TEMPERATURE
    vec_auc = roc_auc_score(ty, torch.sigmoid(vec_logits).numpy())
    print(f"  vector AUC = {vec_auc:.4f}  (model AUC = {te_auc:.4f})")
    print("\n  → serving 时 FAISS IndexFlatIP 检索得到 cosine，再除 τ 即 logit")


if __name__ == "__main__":
    main()

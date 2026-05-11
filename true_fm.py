"""
true_fm.py — 完整 FM + 九老师无损向量化
========================================
对照 fm_demo.py（其实是双塔 sum-pooling），这里实现真正的 FM：

  y = w0 + Σ w_i x_i + Σ_{i<j} <v_i, v_j> x_i x_j

特征切成 user 侧 (user_id, occ) 和 item 侧 (item_id, genres)。二阶项展开：

  Σ_{i<j} <v_i,v_j> = cross_user + cross_item + cross_ui
                     ↑ 同侧       ↑ 同侧       ↑ 跨侧

九老师 trick：把 cross_user / cross_item 算成标量塞进向量某一维，
配合 [...,1,...] 占位，使得 <user_vec, item_vec> = 完整 FM 输出（无损）。

  user_vec = [ S_u,  1,                  W_user + cross_user ]   dim+2
  item_vec = [ S_i,  W_item + cross_item, 1                  ]   dim+2

数据集：MovieLens 100k，rating ≥ 4 = 正样本。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from fm_demo import prepare_data, FMDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TrueFM(nn.Module):
    """完整 Factorization Machine，含同侧交叉项。"""

    def __init__(self, n_users, n_items, dim=64, n_occ=21, n_genres=19):
        super().__init__()
        self.dim = dim

        # 二阶 embedding v_i
        self.u_emb = nn.Embedding(n_users, dim)
        self.i_emb = nn.Embedding(n_items, dim)
        self.o_emb = nn.Embedding(n_occ, dim)
        self.g_emb = nn.Embedding(n_genres, dim)

        # 一阶权重 w_i —— 注意 g_w 这次是 per-genre（修了 fm_demo.py 的 bug）
        self.u_w = nn.Embedding(n_users, 1)
        self.i_w = nn.Embedding(n_items, 1)
        self.o_w = nn.Embedding(n_occ, 1)
        self.g_w = nn.Embedding(n_genres, 1)
        self.w0 = nn.Parameter(torch.zeros(1))

        for emb in [self.u_emb, self.i_emb, self.o_emb, self.g_emb]:
            nn.init.normal_(emb.weight, std=0.01)
        for emb in [self.u_w, self.i_w, self.o_w, self.g_w]:
            nn.init.zeros_(emb.weight)

    # ------------------------------------------------------------------
    # 内部辅助：把 batch 拆成 user 侧/item 侧的 (S, sum||v||^2, W)
    # ------------------------------------------------------------------
    def _user_side(self, batch):
        ue = self.u_emb(batch["user"])             # [B, d]
        oe = self.o_emb(batch["occ"])              # [B, d]
        S_u = ue + oe                              # [B, d]
        SS_u = (ue * ue).sum(-1) + (oe * oe).sum(-1)   # [B]
        W_u = self.u_w(batch["user"]).squeeze(-1) + self.o_w(batch["occ"]).squeeze(-1)
        return S_u, SS_u, W_u

    def _item_side(self, batch):
        ie = self.i_emb(batch["item"])             # [B, d]
        mask = batch["genres"].unsqueeze(-1)       # [B, n_g, 1]
        ge = self.g_emb.weight.unsqueeze(0) * mask # [B, n_g, d]
        g_sum = ge.sum(dim=1)                      # [B, d]
        S_i = ie + g_sum                           # [B, d]
        # sum ||v_g||^2 over active genres
        SS_g = ((self.g_emb.weight ** 2).sum(-1).unsqueeze(0) * batch["genres"]).sum(-1)
        SS_i = (ie * ie).sum(-1) + SS_g            # [B]
        W_i = (self.i_w(batch["item"]).squeeze(-1)
               + (self.g_w.weight.squeeze(-1).unsqueeze(0) * batch["genres"]).sum(-1)
               + self.w0)                          # [B]
        return S_i, SS_i, W_i

    # ------------------------------------------------------------------
    # forward —— 显式拆出 cross_user / cross_item / cross_ui，便于校验
    # ------------------------------------------------------------------
    def forward(self, batch):
        S_u, SS_u, W_u = self._user_side(batch)
        S_i, SS_i, W_i = self._item_side(batch)

        cross_user = 0.5 * ((S_u * S_u).sum(-1) - SS_u)
        cross_item = 0.5 * ((S_i * S_i).sum(-1) - SS_i)
        cross_ui   = (S_u * S_i).sum(-1)

        return cross_ui + cross_user + cross_item + W_u + W_i

    # ------------------------------------------------------------------
    # Serving 向量构造（九老师 trick）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build_user_vectors(self, user_ids, occ_ids):
        batch = {"user": user_ids, "occ": occ_ids}
        S_u, SS_u, W_u = self._user_side(batch)
        cross_user = 0.5 * ((S_u * S_u).sum(-1) - SS_u)        # [N]
        n = len(user_ids)
        # [S_u | 1 | W_u + cross_user]
        return torch.cat([
            S_u,
            torch.ones(n, 1),
            (W_u + cross_user).unsqueeze(-1),
        ], dim=1)

    @torch.no_grad()
    def build_item_vectors(self, item_ids, genres_multi_hot):
        batch = {"item": item_ids, "genres": genres_multi_hot}
        S_i, SS_i, W_i = self._item_side(batch)
        cross_item = 0.5 * ((S_i * S_i).sum(-1) - SS_i)        # [N]
        n = len(item_ids)
        # [S_i | W_i + cross_item | 1]
        return torch.cat([
            S_i,
            (W_i + cross_item).unsqueeze(-1),
            torch.ones(n, 1),
        ], dim=1)


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------
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
    print("  TrueFM — 完整 FM (含同侧交叉) + 九老师无损向量化")
    print("=" * 60)

    data = prepare_data()
    print(f"\n  {data['n_users']} users, {data['n_items']} items, "
          f"{len(data['y'])} ratings")

    train_ds = FMDataset(data, data["train"])
    val_ds   = FMDataset(data, data["val"])
    test_ds  = FMDataset(data, data["test"])
    train_loader = DataLoader(train_ds, 1024, shuffle=True)
    val_loader   = DataLoader(val_ds, 1024)
    test_loader  = DataLoader(test_ds, 1024)

    model = TrueFM(data["n_users"], data["n_items"], dim=64,
                   n_occ=data["n_occ"], n_genres=data["n_genres"])
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(optim, 5, 0.5)

    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n  Training (20 epochs)...\n")
    best_val = 0
    for ep in range(20):
        tr_loss, tr_auc = run_epoch(model, train_loader, optim)
        va_loss, va_auc = run_epoch(model, val_loader)
        sched.step()
        mark = ""
        if va_auc > best_val:
            best_val = va_auc
            torch.save(model.state_dict(), "true_fm.pt")
            mark = "  ★"
        print(f"  E{ep+1:2d}  tr={tr_loss:.4f}/{tr_auc:.4f}  "
              f"va={va_loss:.4f}/{va_auc:.4f}{mark}")

    model.load_state_dict(torch.load("true_fm.pt"))
    te_loss, te_auc = run_epoch(model, test_loader)
    print(f"\n  Test  loss={te_loss:.4f}  AUC={te_auc:.4f}")

    # --- 九老师 trick 验证 ---
    print("\n" + "-" * 60)
    print("  Vectorization check：<user_vec, item_vec> 应严格等于完整 FM")
    print("-" * 60)

    # 全量 user / item 向量
    user_occ = torch.zeros(data["n_users"], dtype=torch.long)
    for u in range(data["n_users"]):
        user_occ[u] = data["occs"].get(u, 0)
    g_full = torch.zeros(data["n_items"], data["n_genres"])
    for i in range(data["n_items"]):
        for k in data["item_genres"].get(i, []):
            g_full[i, k] = 1.0

    user_vecs = model.build_user_vectors(torch.arange(data["n_users"]), user_occ)
    item_vecs = model.build_item_vectors(torch.arange(data["n_items"]), g_full)
    print(f"  user_vec={tuple(user_vecs.shape)}  item_vec={tuple(item_vecs.shape)}")

    # 用测试集逐条对比
    tu = torch.tensor(data["u"][data["test"]], dtype=torch.long)
    ti = torch.tensor(data["i"][data["test"]], dtype=torch.long)
    ty = data["y"][data["test"]]
    with torch.no_grad():
        # 模型 forward
        test_batch = next(iter(DataLoader(test_ds, batch_size=len(test_ds))))
        model_logits = model(test_batch)
        # 向量点积
        vec_logits = (user_vecs[tu] * item_vecs[ti]).sum(-1)

    diff = (model_logits - vec_logits).abs().max().item()
    vec_auc = roc_auc_score(ty, torch.sigmoid(vec_logits).numpy())
    print(f"  max |model_logit - vec_dot| = {diff:.2e}  "
          f"({'PASS' if diff < 1e-4 else 'FAIL'})")
    print(f"  vector AUC = {vec_auc:.4f}  (model AUC = {te_auc:.4f})")
    assert diff < 1e-4, "完整 FM 向量化失败"
    print("\n  ✓ 完整 FM 可被无损还原成单次点积 — 这才是九老师 trick 的精髓")


if __name__ == "__main__":
    main()

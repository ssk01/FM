"""
FM Demo — 多快好省
===================
Implements the user-item vectorization trick from 九老师's Zhihu answer.

Core idea:
  score = <u_emb + sum(u_side_embs), i_emb + sum(i_side_embs)>
          + u_bias + occ_bias + i_bias + genre_bias + global_bias

Serving vectors:
  user_vec = [u_emb + occ_emb, u_bias, occ_bias, 1,             1           ]
  item_vec = [i_emb + g_agg,   1,      1,        i_bias + g_bias + global_bias]
  => CTR = sigmoid(dot(user_vec, item_vec))

Dataset: MovieLens 100k (ml-100k) — 943 users, 1682 movies, 100k ratings.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from urllib.request import urlretrieve
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "ml-100k"
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_data():
    if (DATA_DIR / "u.data").exists():
        return
    import zipfile
    print("Downloading ml-100k ...")
    zpath = DATA_DIR.with_suffix(".zip")
    urlretrieve(ML100K_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA_DIR.parent)
    zpath.unlink()


def load_ratings():
    """Return (user_ids, item_ids, labels), label = 1 if rating >= 4."""
    download_data()
    u, i, y = [], [], []
    with open(DATA_DIR / "u.data") as f:
        for line in f:
            parts = line.strip().split("\t")
            u.append(int(parts[0]) - 1)
            i.append(int(parts[1]) - 1)
            y.append(1 if int(parts[2]) >= 4 else 0)
    return np.array(u), np.array(i), np.array(y, dtype=np.float32)


def load_meta():
    download_data()
    occ_names = [l.strip() for l in open(DATA_DIR / "u.occupation")]
    occ2id = {n: i for i, n in enumerate(occ_names)}

    occs = {}
    with open(DATA_DIR / "u.user") as f:
        for line in f:
            p = line.strip().split("|")
            occs[int(p[0]) - 1] = occ2id[p[3]]

    titles = {}
    item_genres = {}
    with open(DATA_DIR / "u.item", encoding="latin-1") as f:
        for line in f:
            p = line.strip().split("|")
            mid = int(p[0]) - 1
            titles[mid] = p[1]
            item_genres[mid] = [i for i, v in enumerate(p[5:]) if v == "1"]
    return occs, titles, item_genres


def prepare_data(seed=42):
    u, i, y = load_ratings()
    n_users = int(u.max()) + 1
    n_items = int(i.max()) + 1

    np.random.seed(seed)
    perm = np.random.permutation(len(y))
    n_train = int(0.8 * len(y))
    n_val = int(0.1 * len(y))
    occs, titles, item_genres = load_meta()
    return {
        "n_users": n_users, "n_items": n_items, "n_genres": 19, "n_occ": 21,
        "u": u, "i": i, "y": y,
        "train": perm[:n_train], "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
        "occs": occs, "titles": titles, "item_genres": item_genres,
    }


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------
class FMDataset(Dataset):
    def __init__(self, data, indices):
        self.u = data["u"][indices]
        self.i = data["i"][indices]
        self.y = data["y"][indices]
        self.occs = data["occs"]
        self.item_genres = data["item_genres"]

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        uid, iid = int(self.u[idx]), int(self.i[idx])
        occ = self.occs.get(uid, 0)
        g_idx = self.item_genres.get(iid, [])
        g = np.zeros(19, dtype=np.float32)
        g[g_idx] = 1.0
        return {
            "user":   torch.tensor(uid, dtype=torch.long),
            "item":   torch.tensor(iid, dtype=torch.long),
            "occ":    torch.tensor(occ, dtype=torch.long),
            "genres": torch.tensor(g),
            "label":  torch.tensor(self.y[idx], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# 3. Model  (多快好省的核心)
# ---------------------------------------------------------------------------
class FM(nn.Module):
    """Factorization Machine with user-item vectorization support."""

    def __init__(self, n_users, n_items, dim=64, n_occ=21, n_genres=19):
        super().__init__()
        self.dim = dim
        self.n_occ = n_occ
        self.n_genres = n_genres

        # Second-order embeddings
        self.u_emb = nn.Embedding(n_users, dim)
        self.i_emb = nn.Embedding(n_items, dim)
        self.o_emb = nn.Embedding(n_occ, dim)
        self.g_emb = nn.Embedding(n_genres, dim)

        # First-order biases
        self.u_bias = nn.Embedding(n_users, 1)
        self.i_bias = nn.Embedding(n_items, 1)
        self.o_bias = nn.Embedding(n_occ, 1)
        self.g_bias = nn.Parameter(torch.zeros(1))
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.init_weights()

    def init_weights(self):
        for p in [self.u_emb, self.i_emb, self.o_emb, self.g_emb]:
            nn.init.normal_(p.weight, std=0.01)
        for p in [self.u_bias, self.i_bias, self.o_bias]:
            nn.init.zeros_(p.weight)

    def genre_agg(self, genres):
        """Weighted average / sum of genre embeddings."""
        mask = genres.unsqueeze(-1)
        g = self.g_emb.weight.unsqueeze(0) * mask
        return g.sum(dim=1)  # [B, dim]

    def forward(self, batch):
        """Logit = <u_side, i_side> + u_bias + o_bias + i_bias + g_bias + global_bias."""
        u = self.u_emb(batch["user"]) + self.o_emb(batch["occ"])
        i = self.i_emb(batch["item"]) + self.genre_agg(batch["genres"])
        pair = (u * i).sum(dim=1)
        bias = (self.u_bias(batch["user"]).squeeze(-1)
                + self.o_bias(batch["occ"]).squeeze(-1)
                + self.i_bias(batch["item"]).squeeze(-1)
                + self.g_bias + self.global_bias)
        return pair + bias

    # --------------------------------------------------------------
    # Serving vector construction — the key trick
    # --------------------------------------------------------------
    @torch.no_grad()
    def build_user_vectors(self, user_ids, occ_ids):
        """user_vec = [u_emb + o_emb, u_bias, o_bias, 1, 1]  → dim = d + 4"""
        ue = self.u_emb(user_ids) + self.o_emb(occ_ids)
        ub = self.u_bias(user_ids)
        ob = self.o_bias(occ_ids)
        n = len(user_ids)
        return torch.cat([
            ue, ub, ob,
            torch.ones(n, 1),
            torch.ones(n, 1),
        ], dim=1)

    @torch.no_grad()
    def build_item_vectors(self, item_ids, genre_lists):
        """item_vec = [i_emb + g_agg, 1, 1, i_bias, g_bias + global_bias]  → dim = d + 4"""
        ie = self.i_emb(item_ids)
        d = self.dim
        n = len(item_ids)
        g_agg = torch.zeros(n, d)
        for idx, gids in enumerate(genre_lists):
            if gids:
                g_agg[idx] = self.g_emb.weight[gids].sum(dim=0)
        ib = self.i_bias(item_ids)
        gb = (self.g_bias + self.global_bias).expand(n, 1)
        return torch.cat([
            ie + g_agg,
            torch.ones(n, 1),
            torch.ones(n, 1),
            ib, gb,
        ], dim=1)

    @torch.no_grad()
    def ctr(self, user_vecs, item_vecs):
        """CTR = sigmoid(dot(user_vec, item_vec))."""
        return torch.sigmoid((user_vecs * item_vecs).sum(dim=1))


# ---------------------------------------------------------------------------
# 4. Training
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
    loss = total_loss / len(loader)
    auc = roc_auc_score(labels.numpy(), torch.sigmoid(logits).numpy())
    return loss, auc


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 58)
    print("  FM Demo  —  多快好省  Factorization Machine")
    print("=" * 58)

    # --- Load ---
    print("\n[1] Loading MovieLens 100k ...")
    data = prepare_data()
    print(f"    {data['n_users']} users, {data['n_items']} items, "
          f"{len(data['y'])} ratings")

    # Maps
    user_occ = torch.zeros(data["n_users"], dtype=torch.long)
    for u in range(data["n_users"]):
        user_occ[u] = data["occs"].get(u, 0)
    item_genre_list = [data["item_genres"].get(i, [])
                       for i in range(data["n_items"])]

    # --- Datasets ---
    train_ds = FMDataset(data, data["train"])
    val_ds   = FMDataset(data, data["val"])
    test_ds  = FMDataset(data, data["test"])
    train_loader = DataLoader(train_ds, 1024, shuffle=True)
    val_loader   = DataLoader(val_ds,   1024)
    test_loader  = DataLoader(test_ds,  1024)

    # --- Model ---
    print("\n[2] Creating FM (embed_dim=64) ...")
    model = FM(data["n_users"], data["n_items"], dim=64,
               n_occ=data["n_occ"], n_genres=data["n_genres"])
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 5, 0.5)

    # --- Train ---
    print("\n[3] Training (20 epochs) ...")
    best_auc = 0
    for epoch in range(20):
        tr_loss, tr_auc = run_epoch(model, train_loader, optim)
        val_loss, val_auc = run_epoch(model, val_loader)
        scheduler.step()
        print(f"    E{epoch+1:2d}  train_loss={tr_loss:.4f} auc={tr_auc:.4f}  "
              f"val_loss={val_loss:.4f} auc={val_auc:.4f}", end="")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), DATA_DIR.parent / "fm_model.pt")
            print("  ★")
        else:
            print()

    # --- Test ---
    print("\n[4] Test ...")
    model.load_state_dict(torch.load(DATA_DIR.parent / "fm_model.pt"))
    test_loss, test_auc = run_epoch(model, test_loader)
    print(f"    Test loss={test_loss:.4f}  AUC={test_auc:.4f}")

    # --- Build vectors ---
    print("\n[5] Building serving vectors ...")
    user_vecs = model.build_user_vectors(
        torch.arange(data["n_users"]), user_occ)
    item_vecs = model.build_item_vectors(
        torch.arange(data["n_items"]), item_genre_list)
    dim = user_vecs.shape[1]
    print(f"    user_vec: {user_vecs.shape}  item_vec: {item_vecs.shape}")
    print(f"    (second-order {model.dim}d + 4 auxiliary = {dim}d)")

    # Verify vectors produce same AUC
    tu = data["u"][data["test"]]
    ti = data["i"][data["test"]]
    ty = data["y"][data["test"]]
    with torch.no_grad():
        vec_ctr = model.ctr(user_vecs[tu], item_vecs[ti])
    vec_auc = roc_auc_score(ty, vec_ctr.numpy())
    match = "✓" if abs(vec_auc - test_auc) < 0.005 else "△"
    print(f"    Vector AUC={vec_auc:.4f}  (model AUC={test_auc:.4f})  {match}")

    # ================================================================
    # Demo A: Ranking (粗排)
    # ================================================================
    print("\n" + "=" * 58)
    print("  Demo A: Ranking (粗排)")
    print("=" * 58)
    demo_uid = 42
    with torch.no_grad():
        uv = user_vecs[demo_uid: demo_uid + 1]
        ctrs = model.ctr(uv, item_vecs).squeeze(0)
        top_vals, top_ids = ctrs.topk(10)

    titles = data["titles"]
    print(f"  User {demo_uid+1} — Top-10 (CTR):")
    for rk, idx in enumerate(top_ids.tolist()):
        print(f"    {rk+1}. [{idx+1}] {titles.get(idx, '?')}  "
              f"ctr={ctrs[idx]:.4f}")

    # ================================================================
    # Demo B: Retrieval via FAISS (向量召回)
    # ================================================================
    print("\n" + "=" * 58)
    print("  Demo B: Vector Retrieval (向量召回)")
    print("=" * 58)
    import faiss

    item_np = item_vecs.numpy().astype(np.float32)
    print(f"  Building FAISS IndexFlatIP (dim={dim}, {data['n_items']} items) ...")
    index = faiss.IndexFlatIP(dim)
    index.add(item_np)
    print(f"  Index size: {index.ntotal}")

    uv_np = user_vecs[demo_uid: demo_uid + 1].numpy().astype(np.float32)
    faiss_scores, faiss_ids = index.search(uv_np, 10)

    print(f"  User {demo_uid+1} — FAISS Top-10 (dot product = logit):")
    for rk in range(10):
        mid = faiss_ids[0, rk]
        logit = faiss_scores[0, rk]
        ctr = 1.0 / (1.0 + np.exp(-logit))
        print(f"    {rk+1}. [{mid+1}] {titles.get(mid, '?')}  "
              f"logit={logit:.4f}  ctr={ctr:.4f}")

    # Verify match
    assert top_ids.tolist() == faiss_ids[0].tolist(), \
        "FAISS results differ from model!"
    print("\n  ✓ FAISS top-K matches model — vectorization is correct!")

    # ================================================================
    print("\n" + "=" * 58)
    print("  Summary — 多快好省")
    print("=" * 58)
    print("""
  多 (Volume)  粗排上万物品从容不迫; FAISS 检索百万级池
  快 (Speed)   内积 ONP 运算, 在线 RT 极短
  好 (Quality) FM 效果不俗 (ml-100k AUC ~{:.4f})
  省 (Cost)    无需复杂在线服务, KV 存储 + FAISS 即可
    """.format(test_auc))


if __name__ == "__main__":
    main()

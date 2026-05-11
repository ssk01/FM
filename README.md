# MF / FM / 双塔 — 向量化召回/粗排/排序 Demo

本仓库从 Matrix Factorization 出发，对照实现四种模型，重点演示**九老师讲的"完整 FM 也能无损向量化"**这一 trick：

| 模型 | 二阶交叉范围 | Loss | 向量化 |
|---|---|---|---|
| `mf.py` | user × item | BCE | dot |
| `fm_demo.py` | 仅跨侧 sum-pool（≠ 真 FM） | BCE | dot |
| `true_fm.py` | **完整 FM**（含同侧交叉） | BCE | **无损** dot（九老师 trick） |
| `two_tower_norm.py` | 跨侧 sum-pool | BCE | cosine / τ=0.07 |
| `two_tower_sampled_softmax.py` | 跨侧 sum-pool | in-batch sampled softmax | cosine / τ=0.07 |

> 警告：`fm_demo.py` 的命名其实是双塔 sum-pool 模型，**不是真 FM**——它只算了跨侧交叉，丢了 ⟨user_id, occ⟩、⟨genre_a, genre_b⟩ 等同侧两两交叉。`true_fm.py` 把这两项也补回来，并演示如何把它们当成标量塞进 user_vec / item_vec 的某一维，实现 `<u_vec, i_vec> ≡ 完整FM输出`（max diff < 1e-6）。

## 数据集

[MovieLens 100k](https://grouplens.org/datasets/movielens/100k/) — 943 用户 × 1682 电影，共 100000 条评分（1-5 分）。

评分 ≥ 4 视为正样本（喜欢），< 4 视为负样本（不喜欢）。随机打乱后 80% 训练 / 10% 验证 / 10% 测试。

### 三份元数据

| 变量 | 来源 | 含义 | 示例 |
|---|---|---|---|
| `occs` | `u.user` 第 4 列 | 用户职业 → ID | technician → 17 |
| `titles` | `u.item` 第 2 列 | 电影标题 | Toy Story (1995) |
| `item_genres` | `u.item` 第 6-24 列 | 19 个 genre flag → 索引列表 | [3,4,5] |

数据行示例：
```
u.user:  1|24|M|technician|85711
u.item:  1|Toy Story (1995)|...|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
                                              ↑ Ani  ↑Chil  ↑Com
```

## 消融实验——各 feature 贡献多少

固定训练流程，仅改变 user/item 侧特征，在 best validation epoch 评估 test AUC：

| 模型 | 用户侧 | 物品侧 | Test AUC | Δ vs MF |
|---|---|---|---|---|
| MF | user_id | item_id | 0.7278 | — |
| MF+occ | user_id + occ | item_id | 0.7331 | +0.005 |
| MF+genre | user_id | item_id + genre | 0.7389 | +0.011 |
| MF+occ+genre | user_id + occ | item_id + genre | 0.7430 | +0.015 |

结论：
- **genre 贡献最大（+0.011）**，是真正的 side feature 信号
- **occ 贡献微弱（+0.005）**，user_id 已隐式编码了职业信息
- **两者叠加 +0.015**，略低于独立贡献之和（信息重叠）

## 五个模型 Test AUC

| 模型 | Test AUC | 说明 |
|---|---|---|
| MF (`mf.py`) | 0.7278 | baseline |
| MF+occ+genre (`fm_demo.py`) | 0.7430 | 跨侧 sum-pool，缺同侧交叉 |
| **完整 FM (`true_fm.py`)** | **0.7829** | 补回 ⟨u,o⟩、⟨g_a,g_b⟩，+0.04 |
| 双塔 norm+τ (`two_tower_norm.py`) | 0.7501 | cosine / 0.07 + BCE |
| 双塔 sampled softmax | 0.6017 | 优化的是 retrieval ranking，AUC 与上面不可比 |

## 评价指标 — AUC

将测试集每个 (user, item) 输入模型得到 logit（`sigmoid` 后为 CTR），与真实 label 计算 ROC-AUC。

AUC 含义：随机抽一个正样本、一个负样本，模型把正样本排在前面的概率。随机模型 AUC=0.5，理想模型 AUC=1.0。

## FM 的两个核心 trick

### Trick 1 — Rendle 2010：O(k·n) 算两两交叉

朴素地写 FM 二阶项是 O(n²)：

```
2nd_order = Σ_{i<j} <v_i, v_j>·x_i·x_j
```

n=10⁶ 特征 × k=64 维 embedding，一条样本要 6×10¹³ 次乘加，**跑不动**。

用平方差恒等式：

```
||a + b||²  =  ||a||² + ||b||² + 2·<a, b>
```

n 个向量推广：

```
|| Σᵢ xᵢ·vᵢ ||²  =  Σᵢ xᵢ²·||vᵢ||²  +  2 · Σ_{i<j} xᵢ·xⱼ·<vᵢ, vⱼ>
```

移项得到 FM 的核心公式：

```
Σ_{i<j} xᵢ·xⱼ·<vᵢ, vⱼ>  =  0.5 · ( || Σᵢ xᵢ·vᵢ ||²  -  Σᵢ xᵢ²·||vᵢ||² )
                                  ↑ S 求一次             ↑ 自己平方加一次
                                  O(k·nnz)              O(k·nnz)
```

复杂度从 **O(k·n²) → O(k·nnz)**（nnz = 非零特征数 ~50）。100w 特征一条样本 3000 次乘加搞定，**快 10⁹ 倍**，且**结果完全等价、不是近似**。

`true_fm.py:99` 就是这两行：
```python
S_i  = ie + g_sum                           # Σ x_i·v_i  (item 侧)
SS_i = (ie*ie).sum(-1) + ((g_emb**2).sum(-1) * genres).sum(-1)
cross_item = 0.5 * ((S_i*S_i).sum(-1) - SS_i)
```

### Trick 2 — 九老师：完整 FM 也能压成单次点积

把 Trick 1 拿到的两两交叉按"在哪一侧"分成 3 块：

```
Σ_{i<j} = Σ_{i<j ∈ U} + Σ_{i<j ∈ I} + Σ_{i∈U, j∈I}
        = cross_user   + cross_item   + cross_ui
```

跨侧那块本来就能化成 `<S_u, S_i>`。**关键 insight：cross_user 和 cross_item 都是标量**——把它们当成一维"附加分"塞进 user_vec / item_vec：

```
user_vec = [ S_u, 1,                  W_user + cross_user ]    维度 = k + 2
item_vec = [ S_i, W_item + cross_item, 1                  ]    维度 = k + 2
```

点积展开 → `<S_u, S_i>` + `1·(W_item + cross_item)` + `(W_user + cross_user)·1` = **完整 FM 输出**，一字不差。

实测 (true_fm.py)：
```
max |model_logit - vec_dot|  =  9.5e-7   ← 浮点误差量级
vector AUC  =  0.7829   (model AUC = 0.7829)
```

### 四重用途（trick 2 解锁的）

| 用法 | 在线方式 | 候选规模 |
|------|----------|----------|
| **CTR 预估（排序）** | `sigmoid(dot(user_vec, item_vec))` | 上百 |
| **粗排** | 从 KV 取 item_vec，逐一点积 | 上万 |
| **向量召回** | FAISS IndexFlatIP 内积检索 | 百万级 |

只要离线把每个 user / item 算一次定长向量塞 KV / FAISS，三场全用同一份向量服务，**多快好省**。

## 运行

```bash
pip install torch numpy scikit-learn faiss-cpu

# Pure MF
python3 mf.py

# 双塔 sum-pool（README 警告：命名叫 fm_demo，其实不是真 FM）
python3 fm_demo.py

# 真正的 FM（含同侧交叉）+ 九老师无损向量化 trick
python3 true_fm.py

# L2-normalized 双塔，τ=0.07，BCE
python3 two_tower_norm.py

# 双塔 + in-batch sampled softmax (InfoNCE)，τ=0.07
python3 two_tower_sampled_softmax.py

# 消融实验（4 个模型对比）
python3 ablation.py

# BCEWithLogitsLoss 拆解
python3 bce_demo.py
```

## 项目结构

```
FM/
├── mf.py                            # Pure Matrix Factorization（baseline）
├── mf_feature.py                    # No user ID, only occ（cold-start demo）
├── fm_demo.py                       # 双塔 sum-pool（命名叫 fm，但其实不是 FM）
├── true_fm.py                       # 完整 FM（含同侧交叉）+ 九老师无损向量化
├── two_tower_norm.py                # 双塔 L2-norm + τ=0.07 + BCE
├── two_tower_sampled_softmax.py     # 双塔 + in-batch sampled softmax
├── ablation.py                      # 消融：量化 occ / genre 贡献
├── bce_demo.py                      # 手算 BCEWithLogitsLoss
├── TRAINING_LOGS.md
├── README.md
└── ml-100k/                         # 自动下载
```

## 参考文献

- [FM 最妙的模型 — 九老师](https://www.zhihu.com/question/362190044/answer/945591801)
- Rendle, S. (2010). *Factorization Machines*. ICDM.
- Koren, Y. et al. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.

# MF / MF + Side Features — 向量化召回/粗排/排序 Demo

本仓库从 Matrix Factorization 出发，逐步加入 side features，并应用九老师介绍的向量化 trick（将 user/item 侧特征各自聚合成定长向量），实现一套向量同时服务召回、粗排、排序。

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

## 评价指标 — AUC

将测试集每个 (user, item) 输入模型得到 logit（`sigmoid` 后为 CTR），与真实 label 计算 ROC-AUC。

AUC 含义：随机抽一个正样本、一个负样本，模型把正样本排在前面的概率。随机模型 AUC=0.5，理想模型 AUC=1.0。

## 向量化 Trick（多快好省的核心）

将特征分为 User 侧和 Item 侧，利用内积分配率：

$$
\sum_{i \in U}\sum_{j \in I} \langle \mathbf{v}_i, \mathbf{v}_j \rangle = \left\langle \sum_{i \in U} \mathbf{v}_i,\; \sum_{j \in I} \mathbf{v}_j \right\rangle
$$

把 bias 项也分配到两侧，各得到一个定长向量，CTR 退化为点积：

$$
\text{CTR} = \sigma(\langle \mathbf{u}, \mathbf{i} \rangle)
$$

在线 inference 只需读取两向量做 dot，无需复杂模型计算。

### 四重用途

| 用法 | 在线方式 | 候选规模 |
|------|----------|----------|
| **CTR 预估（排序）** | `sigmoid(dot(user_vec, item_vec))` | 上百 |
| **粗排** | 从 KV 取 item_vec，逐一点积 | 上万 |
| **向量召回** | FAISS 索引 + 内积检索 | 百万级 |

## 运行

```bash
pip install torch numpy scikit-learn faiss-cpu

# Pure MF
python3 mf.py

# MF + Side Features（含 FAISS 召回 Demo）
python3 fm_demo.py

# 消融实验（4 个模型对比）
python3 ablation.py

# BCEWithLogitsLoss 拆解
python3 bce_demo.py
```

## 项目结构

```
FM/
├── mf.py             # Pure Matrix Factorization（baseline）
├── mf_feature.py     # No user ID, only occupation features（cold-start demo）
├── fm_demo.py        # MF + occ + genre，含向量化 serving 和 FAISS 召回 Demo
├── ablation.py       # 消融实验：量化 occ / genre 各自贡献
├── bce_demo.py       # 手算 BCEWithLogitsLoss 的 standalone demo
├── TRAINING_LOGS.md  # 各模型训练日志
├── README.md
└── ml-100k/          # （自动下载）MovieLens 100k
```

## 参考文献

- [FM 最妙的模型 — 九老师](https://www.zhihu.com/question/362190044/answer/945591801)
- Rendle, S. (2010). *Factorization Machines*. ICDM.
- Koren, Y. et al. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.

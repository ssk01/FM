"""
bce_demo.py — 手算 BCEWithLogitsLoss 看清每一层
================================================

目标：理解 nn.functional.binary_cross_entropy_with_logits 在做什么。
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 造数据
# ---------------------------------------------------------------------------
logits = torch.tensor([2.0, -1.5, 0.1, 3.2])   # 模型原始输出（未过 sigmoid）
labels = torch.tensor([1.0,  0.0, 1.0, 0.0])    # 真实标签

print("=" * 65)
print("  数据")
print("=" * 65)
print(f"  logits (模型原始输出): {logits.tolist()}")
print(f"  labels (真实标签 0/1): {labels.tolist()}")

# ---------------------------------------------------------------------------
# 拆解 BCEWithLogitsLoss = Sigmoid + BCELoss
# ---------------------------------------------------------------------------
p = torch.sigmoid(logits)                        # σ(logits) → 概率
eps = 1e-7
# 数值不稳定手动版: -(y log p + (1-y) log(1-p))
loss_manual = -(labels * torch.log(p + eps) + (1 - labels) * torch.log(1 - p + eps))

# PyTorch 官方函数
loss_official = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

print("\n" + "=" * 65)
print("  拆解：BCEWithLogits = Sigmoid -> log -> BCE")
print("=" * 65)
print(f"  sigmoid(logits) = p         = {[f'{v:.6f}' for v in p.tolist()]}")
print(f"  -[y log p + (1-y) log(1-p)] = {[f'{v:.6f}' for v in loss_manual.tolist()]}")
print(f"  F.binary_cross_entropy_...  = {[f'{v:.6f}' for v in loss_official.tolist()]}")
print(f"  两者一致? {torch.allclose(loss_manual, loss_official)}")

# --------------------------------------------------------------------------
# 如果用手算 sigmoid → BCELoss
# --------------------------------------------------------------------------
loss_two_step = F.binary_cross_entropy(torch.sigmoid(logits), labels, reduction="none")
print(f"  sigmoid + BCELoss           = {[f'{v:.6f}' for v in loss_two_step.tolist()]}")
print(f"  一致? {torch.allclose(loss_official, loss_two_step)}")

# ---------------------------------------------------------------------------
# Mean reduction（默认行为）
# ---------------------------------------------------------------------------
loss_mean = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
print(f"\n  reduction='mean' (默认)     = {loss_mean.item():.6f}")

# ---------------------------------------------------------------------------
# 为什么不能分开算：log(0) 问题
# ---------------------------------------------------------------------------
logits_bad = torch.tensor([100.0])              # 极端 logit
p_bad = torch.sigmoid(logits_bad)               # ≈ 1.0
print("\n" + "=" * 65)
print("  极端情况对比")
print("=" * 65)
print(f"  logit = 100 → sigmoid ≈ {p_bad.item():.10f}")
try:
    l = -(1.0 * torch.log(p_bad) + 0.0 * torch.log(1 - p_bad))
    print(f"  手动 log(1 - sigmoid(100)) = {l.item()}")
except Exception as e:
    print(f"  手动 log(1 - sigmoid(100)) 出错: {e}")

l_safe = F.binary_cross_entropy_with_logits(logits_bad, torch.ones(1))
print(f"  BCEWithLogits(100, 1)       = {l_safe.item():.6f}  ← 数值稳定")

# ---------------------------------------------------------------------------
# 在模型中的实际含义
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  在 MF 模型中的含义")
print("=" * 65)
print("""
  logits = global_bias + u_bias + i_bias + <u_emb, i_emb>
           ↑ 任意实数，越大表示 user 越可能喜欢 item

  loss = BCEWithLogits(logits, label)
       = -[label * log(σ(logits)) + (1-label) * log(1-σ(logits))]

  梯度方向：
    label=1: 推高 logits → σ 更接近 1 → loss ↓
    label=0: 压低 logits → σ 更接近 0 → loss ↓

  这正是 MF/FM 学出有意义的 user/item 向量的原因。
""")

print(f"\n  预测概率 = sigmoid(2.0) = {torch.sigmoid(torch.tensor(2.0)).item():.4f}  → 正样本")
print(f"  预测概率 = sigmoid(-1.5) = {torch.sigmoid(torch.tensor(-1.5)).item():.4f} → 负样本")

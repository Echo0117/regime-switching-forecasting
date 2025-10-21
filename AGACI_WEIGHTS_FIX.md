# AgACI权重动态问题诊断与修复

## 问题描述

从生成的图片 `agaci_weights_switches.png` 可以看到：

1. **权重几乎不变化**: 所有gamma值的权重都保持在约0.20附近（uniform initialization）
2. **权重线重叠**: 不同regime switch的权重轨迹几乎完全重叠
3. **无适应性**: 权重在regime switch前后没有明显的调整

## 根本原因

### 问题根源：学习率衰减过于激进

原始BOA实现使用 `eta_t = eta / sqrt(t)` 作为学习率调度：

```python
# 原始实现（在agaci.py第128行）
eta_t = self.eta / np.sqrt(self.t)
```

这导致：

- **t=10时**: eta_t = 0.5 / sqrt(10) ≈ 0.158 (衰减到初始值的31.6%)
- **t=100时**: eta_t = 0.5 / sqrt(100) = 0.05 (衰减到初始值的10%)
- **t=500时**: eta_t = 0.5 / sqrt(500) ≈ 0.022 (衰减到初始值的4.4%)

### 为什么这对regime-switching数据特别糟糕？

1. **Regime switches发生在数据后期**: 当一个regime switch发生在t=200时，学习率已经衰减到原始值的7%
2. **权重几乎冻结**: 即使梯度很大（如0.95），权重的最大变化只有 eta_t * gradient ≈ 0.035 * 0.95 = 0.033
3. **无法适应新regime**: AgACI的核心优势（在线适应）被完全破坏

### 诊断证据

运行 `experiments/diagnose_weights.py` 的结果：

```
Learning rate decay:
  eta_t[0]:   0.500000
  eta_t[10]:  0.150756
  eta_t[50]:  0.070014
  eta_t[99]:  0.050000
  Ratio (t=99 / t=0): 0.100000  # 衰减了90%!

Weight change statistics:
  Max absolute change: 0.036436
  Mean absolute change: 0.004562  # 平均每步只变化0.0046!
```

## 解决方案

### 1. 新增学习率调度策略

在 `AdaptiveConformalPredictionsTimeSeries/agaci.py` 中添加多种学习率调度：

```python
def _compute_learning_rate(self, t: int) -> float:
    """Compute learning rate based on schedule."""
    if self.lr_schedule == 'constant':
        return self.eta  # 推荐用于regime-switching
    elif self.lr_schedule == 'sqrt':
        return self.eta / np.sqrt(t)  # 原始BOA（过于激进）
    elif self.lr_schedule == 'log':
        return self.eta / np.log(t + 1)  # 适度衰减
    elif self.lr_schedule == 'poly025':
        return self.eta / (t ** 0.25)  # 缓慢衰减
    elif self.lr_schedule == 'poly06':
        return self.eta / (t ** 0.6)  # 较快衰减
```

### 2. 学习率调度对比

| 调度策略 | t=10 | t=50 | t=100 | t=500 | 推荐场景 |
|---------|------|------|-------|-------|---------|
| constant | 0.500 | 0.500 | 0.500 | 0.500 | **Regime-switching** |
| sqrt | 0.151 | 0.070 | 0.050 | 0.022 | IID数据 |
| log | 0.201 | 0.127 | 0.108 | 0.080 | 缓慢变化数据 |
| poly025 | 0.275 | 0.187 | 0.158 | 0.106 | 中等非平稳性 |
| poly06 | 0.119 | 0.047 | 0.031 | 0.012 | 快速变化数据 |

### 3. 默认配置

**对于regime-switching数据，强烈推荐使用 `constant` 学习率调度**：

```bash
python experiments/test_agaci.py \
    --problem Electricity \
    --agaci-lr-schedule constant \
    --agaci-eta 0.5
```

## 修改的文件

1. **AdaptiveConformalPredictionsTimeSeries/agaci.py**
   - `BOA.__init__()`: 添加 `lr_schedule` 参数
   - `BOA._compute_learning_rate()`: 新增学习率计算方法
   - `BOA.update()`: 使用新的学习率调度
   - `fit_predict_AgACI()`: 传递 `lr_schedule` 参数
   - `run_agaci()`: 传递 `lr_schedule` 参数

2. **experiments/utils/acp_utils.py**
   - `agaci_intervals()`: 从args读取 `agaci_lr_schedule` 并传递给 `run_agaci()`

3. **experiments/test_agaci.py**
   - 添加命令行参数 `--agaci-lr-schedule`
   - 更新参数配置字符串，包含lr_schedule信息

## 验证修复效果

### 运行诊断脚本

```bash
python experiments/diagnose_weights.py
```

生成诊断图：
- `figures/diagnostics/weight_dynamics_diagnostic.png`: 权重演化、学习率衰减、权重变化幅度
- `figures/diagnostics/learning_rate_comparison.png`: 不同调度策略对比

### 运行完整测试

使用 **constant** 学习率（推荐）：
```bash
python experiments/test_agaci.py --problem Electricity --agaci-lr-schedule constant --agaci-eta 0.5
```

使用其他调度（对比实验）：
```bash
# 原始BOA（会看到权重几乎不变）
python experiments/test_agaci.py --problem Electricity --agaci-lr-schedule sqrt --agaci-eta 0.5

# 适度衰减
python experiments/test_agaci.py --problem Electricity --agaci-lr-schedule log --agaci-eta 0.5
```

## 预期改进

使用 `constant` 学习率后，应该看到：

1. ✅ **权重变化明显**: 权重在regime switch前后有清晰的调整
2. ✅ **不同颜色的线分离**: 不同switch的权重轨迹不再重叠
3. ✅ **适应性**: 权重能够快速响应新regime的特征
4. ✅ **Coverage提升**: 特别是在regime switch附近的coverage应该改善

## 理论背景

### 为什么原始BOA使用sqrt(t)衰减？

Bernstein Online Aggregation的理论保证基于：
- **渐近最优性**: 在 **IID** 数据下，sqrt(t) 衰减可以达到最优的后悔界 (regret bound)
- **收敛性**: 保证权重收敛到最优专家

### 为什么regime-switching需要不同策略？

1. **非平稳性**: Regime-switching数据违反IID假设
2. **突然变化**: Regime switch是突然的、不连续的
3. **持续适应**: 需要在整个时间序列中保持适应能力
4. **探索-利用权衡**: Constant学习率保持了持续的探索能力

### 理论权衡

| 特性 | sqrt(t) 衰减 | Constant |
|------|-------------|----------|
| IID数据下的后悔界 | ✅ O(√T) | ❌ O(T) |
| Regime-switching适应性 | ❌ 衰减太快 | ✅ 持续适应 |
| 理论收敛保证 | ✅ 强保证 | ⚠️ 较弱 |
| 实践表现（非平稳） | ❌ 差 | ✅ 好 |

## 参考文献

1. **原始BOA**: Wintenberger, O. (2017). "Optimal learning with Bernstein online aggregation." Machine Learning, 106(1), 119-141.
2. **Adaptive learning rates**: Duchi, J., Hazan, E., & Singer, Y. (2011). "Adaptive subgradient methods for online learning and stochastic optimization." JMLR, 12(7).
3. **Non-stationary bandits**: Garivier, A., & Moulines, E. (2011). "On upper-confidence bound policies for switching bandit problems." ALT 2011.

## 总结

**核心发现**: AgACI原始实现的学习率衰减（eta_t = eta / sqrt(t)）对regime-switching数据过于激进，导致权重在数据后期几乎无法更新。

**核心修复**: 引入灵活的学习率调度，默认使用 `constant` 调度以保持持续的适应能力。

**使用建议**:
- **Regime-switching数据**: 使用 `--agaci-lr-schedule constant`
- **IID或缓慢变化数据**: 可以尝试 `log` 或 `poly025`
- **快速对比**: 始终用 `sqrt` 作为baseline来展示改进效果

# 可视化改进说明

## 问题背景

您提出了两个合理的改进需求：

1. **恢复时间指标可视化不够直观**：只有文本输出，难以快速比较
2. **Raw coverage timeline 图太乱**：91个switches，线条密集重叠，难以阅读

## 解决方案

---

## 改进 1: 恢复时间指标的综合可视化

### 原来的问题：
- 只有简单的2面板图（recovery time vs gamma + success rate）
- 没有显示分布信息
- 缺少统计细节

### 新设计：4面板综合视图

#### **面板 1: Recovery Time vs Gamma（带误差棒）**
```
特点：
• 显示 Mean ± Std（error bars）
• 显示 Median（虚线）
• 对数X轴（gamma）方便看趋势
• 数值标签直接显示在点上
```

**解释**：
- Error bars 显示恢复时间的**变异性**
- 大error bar = 不一致（有些switch恢复快，有些慢）
- 小error bar = 稳定（所有switch恢复时间相似）

#### **面板 2: Recovery Success Rate（柱状图）**
```
特点：
• 彩色柱状图，按方法类型着色
• 每个柱上显示百分比数值
• 100%目标线（绿色虚线）
• 清楚显示哪些方法失败率高
```

**解释**：
- 高度接近100% = 可靠
- 低于100% = 有些switches无法恢复

#### **面板 3: Recovery Time Distribution（箱线图）**
```
特点：
• Box plot显示完整分布
• 红线 = 中位数
• 蓝色虚线 = 平均值
• 箱体 = 25%-75%四分位数
• 须线 = 最小-最大（排除异常值）
• 离群点单独标记
```

**为什么有用**：
- 看到**不只是平均值**，还有整个分布
- 识别**异常switches**（outliers）
- 比较**变异性**（箱体大小）

**示例解读**：
```
γ=0.001: 箱体 [10, 15], 中位数=12
→ 大多数switches恢复需要10-15步

γ=0.99: 箱体 [0, 5], 但有outlier在20+
→ 通常很快，但偶尔失控
```

#### **面板 4: Summary Table（汇总表）**
```
列：
• Gamma
• Mean (steps)
• Median (steps)
• Std (steps)
• Success Rate
• Total Switches
• Failed
```

**为什么有用**：
- 快速参考所有数字
- 适合放在论文/报告中
- 专业的外观

### 使用方法

代码保持不变：
```python
recovery_metrics = compute_recovery_metrics(
    intervals_dict, y_true, d_argmax_test,
    target_coverage=0.9,
    recovery_threshold=0.85,
    window_size=10
)

plot_recovery_comparison(
    recovery_metrics,
    save_path="recovery_comparison.png"
)
```

**自动生成4面板布局！**

---

## 改进 2: Coverage Raw Timeline 的清晰可视化

### 原来的问题：
```
❌ 91 个 regime switches
❌ 5-7 条线重叠
❌ Binary (0/1) 跳跃，难以跟踪
❌ 图片体积大（1.7MB）
❌ 很难看出模式
```

示例（原图）：
```
Coverage
  1.0 ┤███████████████████████████████
      │╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲
  0.5 ┤    （太多线重叠！）
      │
  0.0 ┤
      └─────────────────────────────>
           Time (91 switches)
```

### 新设计：双模式可视化

#### **模式 1: Heatmap（默认，推荐）**

**布局**：
```
┌─────────────────────────────────────────┐
│ Regime Sequence (colorbar)              │ ← 顶部面板
│ 0 1 0 1 1 0 1 ...                       │   Regime ID
│ | | | | |  (91 switches marked)         │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┬────┐
│                                         │ Avg│
│ Naive      ████░██░█░░█████░░░█████    │0.68│
│ AgACI      ████████████████████████     │0.86│
│ γ=0.001    ███████░░░█████░░███████     │0.85│
│ γ=0.01     ████████████████████████     │0.88│
│ γ=0.1      ████████████░░█████░░███     │0.87│
│ γ=0.5      ████░░░████████░░░██████     │0.82│
│ γ=0.99     ███░░░░███████░░░░█████      │0.80│
│                                         │    │
│ Green = Covered (1)                     │    │
│ Red   = Missed (0)                      │    │
│ White = NaN                             │    │
└─────────────────────────────────────────┴────┘
   └─ 红色竖线标记regime switches
```

**优势**：
1. ✅ **每个方法独立一行** - 零重叠
2. ✅ **颜色编码** - 绿色=好，红色=差
3. ✅ **垂直对比** - 容易比较不同gamma
4. ✅ **水平模式识别** - 看哪些区域所有方法都失败
5. ✅ **平均值显示** - 右侧直接看到总体performance
6. ✅ **文件小** - 只有210KB（原来1.7MB）

**如何阅读**：
- **垂直看**：某个时间点，哪个方法最好？
- **水平看**：某个方法的coverage模式如何？
- **颜色密度**：绿色越多 = coverage越好

**发现模式**：
```
如果看到：
- 某一列全是红色 → 这个时间点**所有方法**都miss（可能是极端outlier）
- 某一行有红色块 → 这个方法在某些**regime**表现差
- 大gamma的行红色闪烁 → 证实**振荡假设**
```

#### **模式 2: Rolling Average（可选）**

当你想要**更平滑的曲线**时使用：

**特点**：
```
• Rolling window average（默认10步）
• 平滑掉单个miss的噪声
• 仍然是线图，但更易读
• 添加90%目标线
```

**使用场景**：
- Switches较少（< 20）
- 想要看**趋势**而非逐点
- 需要连续曲线（presentation时更smooth）

### 使用方法

**默认（Heatmap模式）**：
```python
plot_coverage_raw_timeline(
    intervals_dict,
    y_true,
    d_argmax_test,
    save_path="coverage_raw_timeline.png"
    # use_heatmap=True 是默认的
)
```

**切换到Rolling Average模式**：
```python
plot_coverage_raw_timeline(
    intervals_dict,
    y_true,
    d_argmax_test,
    save_path="coverage_raw_timeline_smooth.png",
    use_heatmap=False  # ← 改这里
)
```

---

## 对比总结

### Recovery Time Visualization

| 维度 | 旧版本 | 新版本 (4-panel) |
|------|--------|------------------|
| **信息量** | 2个图 | 4个面板 |
| **分布信息** | ❌ 无 | ✅ Box plot |
| **统计表** | ❌ 无 | ✅ 完整表格 |
| **误差/变异性** | ❌ 不显示 | ✅ Error bars |
| **快速参考** | ⚠️ 需要看图 | ✅ 表格一目了然 |
| **专业度** | ⚠️ 基础 | ✅ 出版级别 |

### Coverage Timeline Visualization

| 维度 | 旧版本 (Raw) | 新版本 (Heatmap) | 新版本 (Rolling) |
|------|--------------|------------------|------------------|
| **可读性** | ❌ 很差 | ✅ 优秀 | ✅ 良好 |
| **重叠问题** | ❌ 严重 | ✅ 零重叠 | ⚠️ 轻微 |
| **对比容易度** | ❌ 困难 | ✅ 容易（垂直对比）| ⚠️ 中等 |
| **模式识别** | ❌ 困难 | ✅ 容易 | ⚠️ 中等 |
| **文件大小** | ❌ 1.7MB | ✅ 210KB | ✅ ~300KB |
| **适用场景** | - | 多switches | 少switches |
| **论文/报告** | ❌ 不适合 | ✅ 非常适合 | ✅ 适合 |

---

## 实际应用示例

### 场景 1: 诊断Gamma值选择

**使用Recovery Comparison（4-panel）**：

看Box plot和Table：
```
γ=0.001: Mean=12, Std=5  → 稳定但慢
γ=0.01:  Mean=5,  Std=2  → 快速且稳定 ⭐
γ=0.1:   Mean=2,  Std=1  → 非常快
γ=0.5:   Mean=2,  Std=8  → 快但不稳定（大Std）❌
γ=0.99:  Mean=3,  Std=12 → 极不稳定 ❌
```

**结论**：γ=0.01 最佳（快速+稳定）

### 场景 2: 识别问题区域

**使用Coverage Heatmap**：

观察发现：
```
时间 50-60: 所有方法都是红色
→ 可能是：
  • Extreme outlier
  • Regime switch特别剧烈
  • 数据质量问题
```

**进一步分析**：
```python
# 查看这个区域的实际数据
y_problem = y_true[50:60]
print(f"Range: [{y_problem.min()}, {y_problem.max()}]")
print(f"Mean: {y_problem.mean()}")

# 是否有异常值？
if y_problem.max() > 3*y_true.std():
    print("Found outlier!")
```

### 场景 3: 论文/报告展示

**推荐组合**：

1. **主图**：Coverage Heatmap
   - 放在Results section
   - Caption: "Coverage performance across methods. Green indicates successful coverage."

2. **辅助图**：Recovery Comparison (4-panel)
   - 放在Appendix或Supplementary Materials
   - Caption: "Detailed recovery time analysis."

3. **文本引用**：
   ```
   "As shown in the heatmap (Figure X), ACI with γ=0.01
   achieved the most consistent coverage (88%, green regions),
   while larger gamma values (0.5+) exhibited oscillatory
   behavior (intermittent red regions)."
   ```

---

## 技术细节

### Heatmap颜色方案

使用`RdYlGn`（Red-Yellow-Green）：
```
0.0 (missed)    → Red
0.5 (partial)   → Yellow
1.0 (covered)   → Green
NaN             → White
```

**为什么选这个**：
- ✅ 直观：红=差，绿=好
- ✅ 色盲友好（相对）
- ✅ 打印友好（即使黑白打印也能区分）

### Box Plot解读

```
      │
  25──┤   ┌─────┐     ← 上须线 (Q3 + 1.5*IQR)
      │   │     │
  20──┤   │     │     ← Q3 (75th percentile)
      │   │─────│     ← Median（红线）
  15──┤   │     │
      │   │     │     ← Q1 (25th percentile)
  10──┤   └─────┘
      │       •       ← Outlier
   5──┤   ┬           ← 下须线 (Q1 - 1.5*IQR)
      │
      └───────────────
```

**IQR** = Interquartile Range = Q3 - Q1

**Outlier定义**：
- 小于 Q1 - 1.5×IQR
- 大于 Q3 + 1.5×IQR

---

## 文件输出

### 自动生成的文件

使用`test_agaci.py`时，自动生成：

```
figures/agaci_test/{dataset}_{config}/
├── recovery_comparison.png        ← 4-panel，436KB
└── coverage_raw_timeline.png      ← Heatmap，210KB
```

### 文件大小对比

| 图表 | 旧版本 | 新版本 | 改善 |
|------|--------|--------|------|
| Recovery | ~120KB | 436KB | ⚠️ 增加（但信息量4倍）|
| Coverage | 1.7MB | 210KB | ✅ 减少88% |

**总体**：1.82MB → 646KB（减少64%）

---

## 总结

### ✅ 改进完成

1. **恢复时间指标可视化**
   - ✅ 4面板综合视图
   - ✅ Box plot显示分布
   - ✅ 统计表格
   - ✅ Error bars显示变异性

2. **Raw Coverage Timeline**
   - ✅ Heatmap模式（默认）- 清晰、紧凑
   - ✅ Rolling average模式（可选）- 平滑曲线
   - ✅ 文件大小大幅减小
   - ✅ 更适合论文/报告

### 🎯 关键优势

- **可读性提升**：从"难以阅读"到"一目了然"
- **信息丰富**：更多统计细节和分布信息
- **专业外观**：适合学术出版
- **文件优化**：更小的文件，更快的加载

### 🚀 使用建议

**日常分析**：
- 使用Heatmap查看coverage模式
- 使用4-panel recovery分析gamma选择

**论文撰写**：
- 主图：Heatmap + 统计表格
- 补充：Box plot（如果需要强调分布）

**演示文稿**：
- Heatmap（清晰、直观）
- Rolling average（如果观众更习惯曲线）

### 📦 代码更改

- 修改文件：`experiments/utils/regime_switch_analysis.py`
- 新增行数：+271行
- 删除行数：-66行
- 净增加：+205行

向后兼容：✅ 所有旧代码仍可运行（默认参数）

---

## 附录：快速参考

### 切换可视化模式

```python
# Heatmap模式（推荐，默认）
plot_coverage_raw_timeline(
    intervals_dict, y_true, d_argmax_test,
    save_path="heatmap.png"
)

# Rolling average模式
plot_coverage_raw_timeline(
    intervals_dict, y_true, d_argmax_test,
    save_path="rolling.png",
    use_heatmap=False
)
```

### 调整Rolling Window大小

在`plot_coverage_raw_timeline`函数内部（1405行）：
```python
window_size = 10  # 默认值，可修改
```

更大的window → 更平滑
更小的window → 更多细节

### 自定义颜色方案

在heatmap代码（1372行）：
```python
im_cov = ax2.imshow(..., cmap='RdYlGn')  # 默认

# 可改为：
# cmap='viridis'  - 蓝紫黄色
# cmap='coolwarm' - 蓝白红
# cmap='binary'   - 黑白
```

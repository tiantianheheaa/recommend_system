- 问题
- 分位数目标的常用损失有哪些？ 给出公式，并给出具体例子来计算。   并说明分位数目标的损失 应该如何和二分类目标的损失 融合到一起

--- 

在短视频推荐的精排多任务学习中，分位数目标（Quantile Target）最常用的损失是 **Pinball Loss**（又称 Check Function Loss）。下面分三部分详细说明：常用损失公式、数值计算示例、以及与二分类损失的融合策略。

---

## 一、分位数目标的常用损失

### 1. Pinball Loss（核心损失）

对于单个分位点 $\tau \in (0,1)$，给定真实值 $y$ 和预测值 $\hat{y}_\tau$，定义残差 $u = y - \hat{y}_\tau$：

$$
\mathcal{L}_{\tau}^{\text{pinball}}(y, \hat{y}_\tau) = 
\begin{cases}
\tau \cdot u, & \text{if } u \geq 0 \quad (\text{预测偏低，欠估计}) \\
(\tau - 1) \cdot u, & \text{if } u < 0 \quad (\text{预测偏高，过估计})
\end{cases}
$$

等价紧凑写法：

$$
\mathcal{L}_{\tau} = u \cdot \left(\tau - \mathbb{I}[u < 0]\right) = \max\left(\tau u,\ (\tau-1)u\right)
$$

**直观理解**：$\tau$ 决定了对正负残差的惩罚不对称性。
- 当 $\tau=0.5$ 时，退化为 **MAE**（中位数回归）。
- 当 $\tau=0.9$ 时，模型对**欠预测**（$u>0$）惩罚很重（系数0.9），但对**过预测**（$u<<0$）惩罚很轻（系数0.1）。这迫使模型偏向保守估计，确保90%的样本真实值不超过预测值。

### 2. Huberized Pinball Loss（鲁棒版）

原始 Pinball Loss 在残差很大时梯度恒定，对异常值敏感。结合 Huber Loss 后：

$$
\mathcal{L}_{\tau}^{\text{Huber}}(u) = 
\begin{cases}
\tau \cdot \frac{u^2}{2\delta}, & \text{if } 0 \leq u \leq \delta \\
(\tau-1) \cdot \frac{u^2}{2\delta}, & \text{if } -\delta \leq u < 0 \\
\tau \cdot (u - \frac{\delta}{2}), & \text{if } u > \delta \\
(\tau-1) \cdot (u + \frac{\delta}{2}), & \text{if } u < -\delta
\end{cases}
$$

其中 $\delta$ 为 Huber 阈值（如 $\delta=0.1$）。残差较小时用二次函数（平滑），较大时退化为线性 Pinball Loss。

### 3. Expectile Loss（可替代方案）

Expectile 是对称化的分位数概念，损失为：

$$
\mathcal{L}_{\omega}(u) = |\omega - \mathbb{I}[u < 0]| \cdot u^2
$$

当 $\omega=0.5$ 时就是 MSE。工业界中，Expectile 回归可以通过**加权 MSE** 实现，训练效率高于 Pinball Loss（无需分支判断），但解释性稍弱。

---

## 二、具体数值计算示例

### 场景设定

<img width="1112" height="394" alt="image" src="https://github.com/user-attachments/assets/a81282c1-868e-4231-97e2-ae79ea16208e" />


### 逐步计算

#### 分位点 $\tau = 0.25$，预测 $\hat{y} = 0.30$

残差 $u = y - \hat{y} = 0.60 - 0.30 = 0.30 > 0$（模型欠预测）

$$
\mathcal{L}_{0.25} = \tau \cdot u = 0.25 \times 0.30 = \mathbf{0.075}
$$

#### 分位点 $\tau = 0.50$，预测 $\hat{y} = 0.55$

残差 $u = 0.60 - 0.55 = 0.05 > 0$（模型欠预测）

$$
\mathcal{L}_{0.50} = 0.50 \times 0.05 = \mathbf{0.025}
$$

#### 分位点 $\tau = 0.75$，预测 $\hat{y} = 0.80$

残差 $u = 0.60 - 0.80 = -0.20 < 0$（模型过预测）

$$
\mathcal{L}_{0.75} = (\tau - 1) \cdot u = (-0.25) \times (-0.20) = \mathbf{0.050}
$$

或者用对称形式：$(1-\tau) \cdot |u| = 0.25 \times 0.20 = 0.050$

### 该样本的总分位数损失

对 $K=3$ 个分位点取平均：

$$
\mathcal{L}_{\text{quantile}} = \frac{0.075 + 0.025 + 0.050}{3} = \mathbf{0.050}
$$

### 对比：二分类 BCE 损失计算

同一样本，假设 **点赞（Like）** 真实标签 $y_{\text{like}} = 1$（点了赞），模型预测概率 $\hat{p}_{\text{like}} = 0.85$：

$$
\mathcal{L}_{\text{like}} = -\left[1 \cdot \log(0.85) + 0 \cdot \log(0.15)\right] = -\log(0.85) \approx \mathbf{0.163}
$$

**关键观察**：Pinball Loss 和 BCE Loss 数值都在 $10^{-1} \sim 10^{-2}$ 量级，但背后的**梯度行为**和**样本覆盖度**差异极大：
- 分位数损失：**每个样本都有监督**（稠密梯度），且梯度方向由 $\tau$ 和残差符号共同决定。
- 点赞 BCE：**99% 样本是负例**（稀疏梯度），正例梯度被大量负例淹没。

---

## 三、与二分类损失的融合策略

现有二分类目标：Like、Follow、Play3s、Play7s。将分位数损失 $\mathcal{L}_{\text{quantile}}$ 融入多任务学习，工业界有四种主流策略：

### 策略 1：直接加权融合（Baseline）

$$
\mathcal{L}_{\text{total}} = w_{\text{like}}\mathcal{L}_{\text{like}} + w_{\text{follow}}\mathcal{L}_{\text{follow}} + w_{\text{play3s}}\mathcal{L}_{\text{play3s}} + w_{\text{play7s}}\mathcal{L}_{\text{play7s}} + w_{\text{q}}\mathcal{L}_{\text{quantile}}
$$

**问题**：各任务损失量级天然不同。例如 Play3s 正样本率 30%，BCE 约 0.5；Follow 正样本率 1%，BCE 约 0.06。固定权重难以平衡，且分位数损失随归一化方式不同数值波动大。

### 策略 2：同方差不确定性加权（Kendall et al.，推荐）

为每个任务引入可学习的噪声参数 $\sigma_i$，让模型自动决定各任务权重：

$$
\mathcal{L}_{\text{total}} = \sum_{i \in \{\text{like, follow, play3s, play7s}\}} \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log \sigma_i \quad+\quad \frac{1}{2\sigma_q^2} \mathcal{L}_{\text{quantile}} + \log \sigma_q
$$

**原理**：
- 若某任务（如 Follow）噪声大、损失波动剧烈，模型会自动增大 $\sigma_i$，降低其权重。
- 分位数目标提供稠密稳定的梯度，通常 $\sigma_q$ 会收敛到较小值，自动获得较高有效权重，**缓解二分类稀疏性带来的训练不充分问题**。

### 策略 3：梯度归一化（GradNorm）

动态调整 $w_i(t)$，使得不同任务在共享层上的梯度量级相当：

$$
w_i(t) \propto \left\| \nabla_\theta w_i(t-1) \mathcal{L}_i \right\|^{-\alpha}
$$

其中 $\alpha$ 为超参。分位数损失因为连续且稠密，其梯度范数通常比稀疏二分类任务更稳定，GradNorm 会自动给稀疏任务（如 Follow）分配更大权重以追赶训练进度。

### 策略 4：分阶段融合（工业界实用技巧）

| 训练阶段 | 策略 | 目的 |
|---------|------|------|
| **冷启动期**（前 1-2 epoch） | $w_q = 2.0 \sim 5.0$，二分类 $w=1.0$ | 用分位数目标的稠密梯度快速预热共享表征，避免稀疏二分类任务主导导致模型坍缩 |
| **稳定期** | 切到不确定性加权或 GradNorm | 让模型自动平衡各任务 |
| **微调期** | 冻结共享层，单独微调各塔 | 特别是稀疏二分类塔，需要在其自身数据上精细调整 |

---

## 四、融合时的关键工程细节

| 细节 | 说明 |
|------|------|
| **损失归一化** | 分位数损失输入建议归一化到 $[0,1]$（如 $\min(W, 300)/300$），避免数值过大压制二分类损失 |
| **梯度裁剪** | Pinball Loss 在残差极大时梯度恒定（线性），需配合全局梯度裁剪（如 clip_norm=1.0）防止异常样本冲击共享层 |
| **采样平衡** | 分位数目标无样本不平衡问题；二分类任务（尤其 Follow）建议用**负采样**或**Focal Loss**辅助，避免被分位数任务的梯度淹没 |
| **评估隔离** | 融合训练时，线上评估应**分别看各任务 AUC/MAE**，防止分位数损失下降但二分类 AUC 劣化的"跷跷板"现象 |

### 推荐配置（可直接落地）

```python
# 损失融合代码示意（对应之前提供的模型）
loss_fn = MultiTaskLoss(
    quantiles=[0.25, 0.5, 0.75],
    task_weights={
        'like': 1.0,
        'follow': 1.0, 
        'play3s': 1.0,
        'play7s': 1.0,
        'quantile': 2.0  # 冷启动期给更高权重，稳定期可降至1.0
    }
)
# 内部使用不确定性加权自动调节，task_weights 作为先验偏置
```

**总结**：分位数损失（Pinball Loss）通过不对称惩罚刻画条件分布，与二分类 BCE 融合时，**核心矛盾是梯度密度差异**（稠密 vs. 稀疏）。推荐采用**不确定性加权**或**冷启动期加权预热**策略，让分位数任务的稠密梯度帮助共享层学习，再由二分类任务精细化各目标边界。

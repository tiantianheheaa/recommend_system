- 问题：
- 短视频推荐中，分位数目标的定义、作用，以及如何应用到精排预估目标中。  精排目标已经有点赞、关注、3s播放、7s播放 等二分类目标，如何加入分位数目标？   设计并实现


--- 
在短视频推荐精排中，"分位数目标"（Quantile Target）通常指**分位数回归（Quantile Regression, QR）**或其衍生形式——将观看时长等连续信号建模为条件分布的多分位点估计，而非单点均值估计。这与传统回归（预测 $\mathbb{E}[W|x]$）有本质区别：它预测的是 $P(W \leq q_\tau | x) = \tau$ 中的 $q_\tau$。

---

## 一、分位数目标的定义与作用

### 1. 定义

设观看时长为随机变量 $W$，给定用户-视频特征 $x$，传统方法预测条件期望 $\hat{w} = \mathbb{E}[W|x]$；而分位数目标预测的是条件分布的多个分位点：

$$\hat{w}_\tau = Q_W(\tau|x), \quad \tau \in \{0.1, 0.25, 0.5, 0.75, 0.9\}$$

即：模型输出一组值，表示"在给定 $x$ 下，有 $\tau$ 概率用户的观看时长不超过 $\hat{w}_\tau$"。

### 2. 在精排中的核心作用

| 作用 | 说明 |
|------|------|
| **不确定性建模** | 同一用户对不同视频的行为方差巨大，单点估计会抹平异质性。分位数目标捕捉分布的离散程度（如 IQR），识别"高确定性高价值"与"高风险高潜力"样本。 |
| **去偏标签（Debiasing）** | 将原始观看时长映射到"同视频时长分桶/同用户历史"下的经验分位数，消除视频长度、用户习惯等混淆偏置，得到相对偏好信号。 |
| **精细排序策略** | 不同分位点可对应不同业务策略：低分位数（$\tau=0.1$）用于保守估计保障用户体验，中位数（$\tau=0.5$）用于常规排序，高分位数（$\tau=0.9$）用于探索潜力内容。 |
| **辅助二分类目标** | 作为连续精细信号，为点赞/关注等稀疏二分类目标提供稠密梯度，缓解正样本稀疏问题。 |

---

## 二、如何加入现有精排目标体系

你现有的二分类目标（点赞、关注、3s播放、7s播放）都是**稀疏离散信号**。加入分位数目标有两种主流方案：

### 方案对比

| 方案 | 本质 | 与二分类目标的关系 | 适用场景 |
|------|------|-------------------|---------|
| **A. 多分位数回归任务** | 多任务学习，新增 Quantile Regression Tower | 与二分类塔共享底层（MMoE/PLE），各自独立损失 | 需要预估观看时长分布，做保守/动态策略 |
| **B. 分位数分桶多分类** | 将观看时长转化为去偏后的有序分类标签 | 作为第5个分类塔（如 5-class），与二分类塔并列 | 需要消除时长偏置，直接用于排序 |

下面给出**方案A（多分位数回归）**的完整设计与实现，因为它与现有二分类目标的融合最自然，且能直接输出可解释的分位点供后续策略使用。

---

## 三、设计：多任务精排模型架构

### 3.1 整体架构

```
输入特征 x (用户画像 + 视频特征 + 上下文)
    │
    ▼
┌─────────────────┐
│   Shared Bottom │  (Dense Layers / Transformer)
│   (共享表征层)   │
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┬────────┐
    ▼         ▼        ▼        ▼        ▼
 Like塔   Follow塔  Play3s塔  Play7s塔  Quantile塔
 (二分类)  (二分类)  (二分类)  (二分类)  (多分位回归)
```
<img width="1508" height="962" alt="image" src="https://github.com/user-attachments/assets/d832b9ea-7353-4c41-9371-5f1431a527ee" />


---

## 四、实现：完整 PyTorch 代码

以下代码可直接运行，包含：
1. **分位数标签生成**（含去偏分桶逻辑）
2. **多任务模型**（MMoE + 二分类塔 + Quantile Tower）
3. **联合损失函数**
4. **推理与排序**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

# ==================== 1. 分位数标签生成（去偏版） ====================

class QuantileLabelGenerator:
    """
    生成去偏的分位数标签。
    参考 D2Q / RAD 思路：按视频时长分桶，计算样本在桶内的经验分位数。
    """
    def __init__(self, n_bins: int = 5, duration_edges: List[float] = None):
        """
        n_bins: 分位数分桶数（如5分位）
        duration_edges: 视频时长分桶边界，如 [0, 10, 30, 60, 120, 300]
        """
        self.n_bins = n_bins
        self.duration_edges = duration_edges or [0, 10, 30, 60, 120, 300]
        
    def fit(self, df: List[Dict]):
        """
        基于训练集统计每个时长桶的观看时长CDF。
        df: list of dicts, each with keys 'watch_time', 'video_duration'
        """
        self.cdfs = {}
        for i in range(len(self.duration_edges) - 1):
            low, high = self.duration_edges[i], self.duration_edges[i+1]
            # 取出该时长桶的所有观看时长
            bucket_wt = [d['watch_time'] for d in df 
                        if low <= d['video_duration'] < high]
            if len(bucket_wt) == 0:
                continue
            sorted_wt = np.sort(bucket_wt)
            self.cdfs[(low, high)] = sorted_wt
            
    def transform(self, watch_time: float, video_duration: float) -> Tuple[np.ndarray, float]:
        """
        返回：
        - quantile_label: one-hot 或 ordinal 标签 (n_bins,)
        - quantile_value: 连续分位数值 [0,1]，用于分位数回归
        """
        # 找到对应时长桶
        bucket = None
        for i in range(len(self.duration_edges) - 1):
            low, high = self.duration_edges[i], self.duration_edges[i+1]
            if low <= video_duration < high:
                bucket = (low, high)
                break
        if bucket is None or bucket not in self.cdfs:
            # 默认全局分位
            return np.zeros(self.n_bins), 0.5
        
        sorted_wt = self.cdfs[bucket]
        # 计算经验分位数
        quantile_val = np.searchsorted(sorted_wt, watch_time, side='right') / len(sorted_wt)
        quantile_val = np.clip(quantile_val, 0.0, 1.0)
        
        # 分桶离散标签（用于多分类辅助任务，可选）
        bin_idx = min(int(quantile_val * self.n_bins), self.n_bins - 1)
        quantile_label = np.zeros(self.n_bins)
        quantile_label[bin_idx] = 1.0
        
        return quantile_label, quantile_val


# ==================== 2. 多任务精排模型 ====================

class MMoE(nn.Module):
    """MMoE 共享专家层"""
    def __init__(self, input_dim: int, num_experts: int, num_tasks: int, expert_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for _ in range(num_experts)
        ])
        
        # 门控网络（每个任务一个）
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x: [B, input_dim]
        returns: list of [B, expert_dim] for each task
        """
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, num_experts, expert_dim]
        
        task_inputs = []
        for gate in self.gates:
            weights = gate(x)  # [B, num_experts]
            weighted = torch.einsum('be,bed->bd', weights, expert_outputs)
            task_inputs.append(weighted)
            
        return task_inputs


class ShortVideoRanker(nn.Module):
    """
    短视频精排模型：4个二分类目标 + 1个多分位数回归目标
    """
    def __init__(
        self, 
        feature_dim: int = 128,
        num_experts: int = 4,
        expert_dim: int = 64,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        max_watch_time: float = 300.0  # 时长截断值
    ):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.max_watch_time = max_watch_time
        
        # 任务数：like, follow, play3s, play7s, quantile
        self.task_names = ['like', 'follow', 'play3s', 'play7s', 'quantile']
        self.num_tasks = len(self.task_names)
        
        # MMoE 共享层
        self.mmoelayer = MMoE(feature_dim, num_experts, self.num_tasks, expert_dim)
        
        # 各任务塔
        self.towers = nn.ModuleDict({
            'like': nn.Sequential(nn.Linear(expert_dim, 32), nn.ReLU(), nn.Linear(32, 1)),
            'follow': nn.Sequential(nn.Linear(expert_dim, 32), nn.ReLU(), nn.Linear(32, 1)),
            'play3s': nn.Sequential(nn.Linear(expert_dim, 32), nn.ReLU(), nn.Linear(32, 1)),
            'play7s': nn.Sequential(nn.Linear(expert_dim, 32), nn.ReLU(), nn.Linear(32, 1)),
            # Quantile Tower: 输出 K 个分位点
            'quantile': nn.Sequential(
                nn.Linear(expert_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, self.num_quantiles)  # [B, K]
            )
        })
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, feature_dim]
        returns: dict of predictions
        """
        task_inputs = self.mmoelayer(x)  # list of [B, expert_dim]
        
        outputs = {}
        for i, name in enumerate(self.task_names):
            tower_out = self.towers[name](task_inputs[i])
            if name != 'quantile':
                outputs[name] = torch.sigmoid(tower_out).squeeze(-1)  # [B]
            else:
                # 分位数输出: [B, K]，用 sigmoid 约束到 [0,1] 再映射到 [0, max_watch_time]
                outputs[name] = torch.sigmoid(tower_out) * self.max_watch_time  # [B, K]
                
        return outputs


# ==================== 3. 联合损失函数 ====================

class MultiTaskLoss(nn.Module):
    def __init__(
        self, 
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        task_weights: Dict[str, float] = None,
        max_watch_time: float = 300.0
    ):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
        self.num_quantiles = len(quantiles)
        self.max_watch_time = max_watch_time
        
        # 默认任务权重
        self.task_weights = task_weights or {
            'like': 1.0, 'follow': 1.0, 'play3s': 1.0, 'play7s': 1.0, 'quantile': 1.0
        }
        
        # 可学习的任务不确定性权重（Kendall et al. 多任务同方差不确定性）
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in self.task_weights.keys()
        })
        
    def quantile_loss(
        self, 
        pred: torch.Tensor,  # [B, K]
        target: torch.Tensor  # [B]
    ) -> torch.Tensor:
        """
        分位数损失（Pinball Loss）
        pred: [B, K]  每个分位点的预测
        target: [B]   真实观看时长（已归一化或原始值）
        """
        target = target.unsqueeze(-1)  # [B, 1]
        errors = target - pred  # [B, K]
        
        # quantiles: [K]
        taus = self.quantiles.to(pred.device).view(1, -1)  # [1, K]
        
        # L_tau = max(tau * error, (tau-1) * error)
        losses = torch.max(taus * errors, (taus - 1) * errors)
        return losses.mean()
        
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        targets: {
            'like': [B] (0/1),
            'follow': [B] (0/1),
            'play3s': [B] (0/1),
            'play7s': [B] (0/1),
            'watch_time': [B] (float, 原始观看时长),
            'quantile_bins': [B, n_bins] (可选，用于分桶辅助任务)
        }
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 二分类损失
        for task in ['like', 'follow', 'play3s', 'play7s']:
            pred = predictions[task]  # [B]
            tgt = targets[task].float()  # [B]
            
            bce = F.binary_cross_entropy(pred, tgt, reduction='mean')
            # 不确定性加权
            precision = torch.exp(-self.log_vars[task])
            weighted_loss = precision * bce + self.log_vars[task]
            
            total_loss += self.task_weights[task] * weighted_loss
            loss_dict[task] = bce.item()
            
        # 分位数回归损失
        q_pred = predictions['quantile']  # [B, K]
        q_tgt = targets['watch_time'].clamp(0, self.max_watch_time)  # [B]
        
        q_loss = self.quantile_loss(q_pred, q_tgt)
        precision = torch.exp(-self.log_vars['quantile'])
        weighted_q_loss = precision * q_loss + self.log_vars['quantile']
        
        total_loss += self.task_weights['quantile'] * weighted_q_loss
        loss_dict['quantile'] = q_loss.item()
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# ==================== 4. 训练与推理示例 ====================

def train_step(model, loss_fn, optimizer, batch):
    model.train()
    optimizer.zero_grad()
    
    x = batch['features']  # [B, feature_dim]
    targets = {
        'like': batch['like'],
        'follow': batch['follow'],
        'play3s': batch['play3s'],
        'play7s': batch['play7s'],
        'watch_time': batch['watch_time']
    }
    
    preds = model(x)
    loss, loss_dict = loss_fn(preds, targets)
    loss.backward()
    optimizer.step()
    
    return loss_dict


def inference_and_fuse(
    model: ShortVideoRanker,
    features: torch.Tensor,
    fusion_weights: Dict[str, float] = None
) -> torch.Tensor:
    """
    推理并融合多目标分数
    默认融合：二分类概率 + 中位数分位点（tau=0.5）
    """
    model.eval()
    fusion_weights = fusion_weights or {
        'like': 0.3, 'follow': 0.3, 'play3s': 0.2, 'play7s': 0.1, 
        'quantile_median': 0.1  # 使用 tau=0.5 的预估
    }
    
    with torch.no_grad():
        preds = model(features)
        
        # 取 quantile 的中位数（假设 quantiles=[0.1,0.25,0.5,0.75,0.9]，index=2）
        median_idx = model.num_quantiles // 2
        q_median = preds['quantile'][:, median_idx] / model.max_watch_time  # 归一化到 [0,1]
        
        score = (
            fusion_weights['like'] * preds['like'] +
            fusion_weights['follow'] * preds['follow'] +
            fusion_weights['play3s'] * preds['play3s'] +
            fusion_weights['play7s'] * preds['play7s'] +
            fusion_weights['quantile_median'] * q_median
        )
        
    return score


# ==================== 5. 运行示例 ====================

if __name__ == "__main__":
    # 超参
    BATCH_SIZE = 64
    FEATURE_DIM = 128
    EPOCHS = 5
    
    # 初始化
    model = ShortVideoRanker(
        feature_dim=FEATURE_DIM,
        num_experts=4,
        expert_dim=64,
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    loss_fn = MultiTaskLoss(
        quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
        task_weights={'like':1, 'follow':1, 'play3s':1, 'play7s':1, 'quantile':2.0}
        # quantile 权重可加大，因其提供稠密梯度
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 模拟数据
    for epoch in range(EPOCHS):
        # 模拟一个 batch
        batch = {
            'features': torch.randn(BATCH_SIZE, FEATURE_DIM),
            'like': torch.randint(0, 2, (BATCH_SIZE,)).float(),
            'follow': torch.randint(0, 2, (BATCH_SIZE,)).float(),
            'play3s': torch.randint(0, 2, (BATCH_SIZE,)).float(),
            'play7s': torch.randint(0, 2, (BATCH_SIZE,)).float(),
            'watch_time': torch.rand(BATCH_SIZE) * 300  # 0~300s
        }
        
        loss_dict = train_step(model, loss_fn, optimizer, batch)
        print(f"Epoch {epoch+1}: {loss_dict}")
    
    # 推理
    test_features = torch.randn(10, FEATURE_DIM)
    ranking_scores = inference_and_fuse(model, test_features)
    print("Ranking scores:", ranking_scores)
```

---

## 五、关键设计要点总结

| 设计点 | 说明 |
|--------|------|
| **去偏标签生成** | 按视频时长分桶计算经验分位数（D2Q/RAD思路），消除"长视频天然时长高"的偏置，使标签反映真实兴趣而非时长混淆。 |
| **MMoE 共享架构** | 分位数任务与二分类任务共享底层，但分位数损失提供**稠密梯度**（每个样本都有连续监督），可缓解点赞/关注等稀疏目标的训练信号不足问题。 |
| **多分位点输出** | 同时预测 $\tau \in \{0.1, 0.5, 0.9\}$ 等，线上可灵活组合：新用户/冷启视频用低分位保守估计，活跃用户用中位数，探索场景用高分位。 |
| **不确定性加权** | 使用可学习的同方差不确定性权重（Kendall et al.），自动平衡分位数回归与二分类任务的梯度量级，避免某一方主导。 |
| **排序融合** | 精排最终分可简单线性融合，也可训练一个轻量级融合层（如GBDT或注意力机制），将多分位点作为 richer feature 输入。 |

如需进一步将分位数目标扩展为**方案B（分桶多分类）**或引入**CQE 的条件分布建模**（预测完整 CDF 而非孤立分位点），可以在此基础上继续迭代。

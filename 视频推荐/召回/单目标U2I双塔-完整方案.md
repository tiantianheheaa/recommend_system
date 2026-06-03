 

以下是针对**抖音/快手短视频场景**的 u2i 双塔召回模型的完整设计与实现方案，涵盖 Label 定义、双塔结构、特征输入、正负样本选择等关键环节。

---

## 一、Label 定义

短视频推荐中，双塔召回的 Label 设计直接影响模型学习的目标。业界主流有两种方案：

### 方案 A：点击/完播作为正样本（最常用）

| 类型 | 定义 | 说明 |
|------|------|------|
| **正样本** | 曝光且**有效播放**（如 3s 以上）或**完播** | 短视频场景下，单纯点击不够，需结合播放行为。通常以"3s 播放"或"完播"作为正样本门槛，过滤误触 |
| **负样本** | 见下文"正负样本选择"章节 | |

**为什么不用点赞/关注作为正样本？**
- 点赞/关注极度稀疏（正样本率 < 1%），作为召回正样本会导致训练信号过弱
- 召回的目标是"找到用户可能感兴趣的内容"，有效播放已足够表达兴趣；点赞/关注留给精排做更精细的区分

### 方案 B：观看时长作为连续 Label（进阶）

将观看时长做归一化（如 $\min(W, 300)/300$），作为连续 Label 训练。损失函数可用加权 BCE（权重与时长正相关）或分位数损失。

```python
# 连续 Label 示例
watch_time = min(actual_watch_time, 300) / 300.0  # 截断归一化到 [0,1]
label_weight = 1.0 + 5.0 * watch_time  # 长视频权重更高
```

**工业界共识**：短视频场景以**有效播放/完播作为二分类正样本**为主流，实现简单、效果稳定。连续 Label 可作为实验方向，但需配合更复杂的损失设计。

---

## 二、双塔结构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     双塔召回模型 (Two-Tower)                   │
├─────────────────────────┬───────────────────────────────────┤
│        用户塔 (User Tower) │        视频塔 (Item Tower)        │
│                         │                                   │
│  ┌─────────────────┐   │   ┌─────────────────────────┐    │
│  │ 用户特征输入      │   │   │ 视频特征输入             │    │
│  │ • 用户ID Embedding│   │   │ • 视频ID Embedding       │    │
│  │ • 用户画像特征    │   │   │ • 作者ID Embedding       │    │
│  │ • 行为序列特征    │   │   │ • 视频内容特征 (Tag/类目) │    │
│  │ • 上下文特征      │   │   │ • 视频统计特征             │    │
│  └────────┬────────┘   │   └───────────┬─────────────┘    │
│           ▼             │               ▼                  │
│  ┌─────────────────┐   │   ┌─────────────────────────┐    │
│  │ MLP / Transformer│   │   │ MLP                     │    │
│  │ (64→128→64)     │   │   │ (64→128→64)             │    │
│  └────────┬────────┘   │   └───────────┬─────────────┘    │
│           ▼             │               ▼                  │
│    User Vector (64-dim) │    Item Vector (64-dim)         │
│           │             │               │                  │
└───────────┼─────────────┴───────────────┼──────────────────┘
            │                             │
            └────────────┬────────────────┘
                         ▼
              Similarity = Cosine(u, v) 或 Inner Product
                         ▼
              Sigmoid(CrossEntropy) / Softmax
```

### 2.2 用户塔详细设计

| 特征类别 | 具体特征 | 处理方式 | 维度 |
|---------|---------|---------|------|
| **用户 ID** | user_id | Embedding (Hash 到 100w 桶) | 32 |
| **用户画像** | 性别、年龄段、城市等级、设备类型 | Embedding / One-hot | 各 8-16 |
| **长期兴趣** | 近 30 天高互动类目 Top-N | Embedding Pooling (Sum/Avg) | 32 |
| **短期行为序列** | 近 50 条观看视频 ID 序列 | ID Embedding + Self-Attention / DIN | 32 |
| **上下文** | 请求时间（小时/星期）、网络类型 | Embedding / 连续值归一化 | 8-16 |

**关键设计：行为序列建模**

短视频用户兴趣漂移快，短期序列比长期统计更重要：

```python
# 用户塔核心代码
class UserTower(nn.Module):
    def __init__(self, user_id_vocab, video_id_vocab, embed_dim=32, seq_len=50):
        super().__init__()
        self.user_embed = nn.Embedding(user_id_vocab, embed_dim)
        self.video_embed = nn.Embedding(video_id_vocab, embed_dim)
        
        # 行为序列 Self-Attention
        self.seq_attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.seq_fc = nn.Linear(embed_dim, embed_dim)
        
        # 画像特征
        self.profile_fc = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2)
        )
        
        # 输出层
        self.output_fc = nn.Sequential(
            nn.Linear(32 + 32 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward(self, user_id, video_seq, seq_mask, profile_features):
        # user_id: [B]
        # video_seq: [B, seq_len] 最近观看的视频ID
        # seq_mask: [B, seq_len] padding mask
        # profile_features: [B, 64]
        
        u_emb = self.user_embed(user_id)  # [B, 32]
        
        # 序列建模
        seq_emb = self.video_embed(video_seq)  # [B, seq_len, 32]
        seq_emb, _ = self.seq_attention(seq_emb, seq_emb, seq_emb, 
                                        key_padding_mask=~seq_mask.bool())
        seq_repr = self.seq_fc(seq_emb.mean(dim=1))  # [B, 32]
        
        # 画像
        profile_repr = self.profile_fc(profile_features)  # [B, 64]
        
        # 融合
        concat = torch.cat([u_emb, seq_repr, profile_repr], dim=-1)
        user_vec = F.normalize(self.output_fc(concat), dim=-1)  # [B, 64], L2归一化
        
        return user_vec
```

### 2.3 视频塔详细设计

| 特征类别 | 具体特征 | 处理方式 | 维度 |
|---------|---------|---------|------|
| **视频 ID** | video_id | Embedding (Hash 到 1000w 桶) | 32 |
| **作者 ID** | author_id | Embedding | 16 |
| **内容理解** | 类目、标签、BGM ID、文案关键词 | Multi-hot Embedding + Pooling | 32 |
| **统计特征** | 近 7 天播放量、点赞率、完播率、发布时长 | 连续值 + 分桶离散化 | 16 |
| **多模态预训练** | 封面图向量、视频关键帧向量 (可选) | 预训练模型输出，冻结或微调 | 64 |

```python
class ItemTower(nn.Module):
    def __init__(self, video_id_vocab, author_id_vocab, tag_vocab, embed_dim=32):
        super().__init__()
        self.video_embed = nn.Embedding(video_id_vocab, embed_dim)
        self.author_embed = nn.Embedding(author_id_vocab, 16)
        self.tag_embed = nn.Embedding(tag_vocab, 16)
        
        # 统计特征处理
        self.stat_fc = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU()
        )
        
        self.output_fc = nn.Sequential(
            nn.Linear(32 + 16 + 32 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward(self, video_id, author_id, tags, tag_mask, stat_features):
        v_emb = self.video_embed(video_id)  # [B, 32]
        a_emb = self.author_embed(author_id)  # [B, 16]
        
        # 标签多热编码
        tag_emb = self.tag_embed(tags)  # [B, max_tags, 16]
        tag_emb = (tag_emb * tag_mask.unsqueeze(-1)).sum(dim=1)  # [B, 16]
        tag_emb = tag_emb / (tag_mask.sum(dim=1, keepdim=True) + 1e-8)
        
        stat_repr = self.stat_fc(stat_features)  # [B, 32]
        
        concat = torch.cat([v_emb, a_emb, tag_emb, stat_repr], dim=-1)
        item_vec = F.normalize(self.output_fc(concat), dim=-1)  # [B, 64]
        
        return item_vec
```

### 2.4 相似度计算与损失

```python
class TwoTowerModel(nn.Module):
    def __init__(self, user_tower, item_tower, temperature=0.05):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.temperature = temperature  # 温度系数，控制分布锐度
        
    def forward(self, user_features, item_features):
        user_vec = self.user_tower(**user_features)   # [B, 64]
        item_vec = self.item_tower(**item_features)   # [B, 64]
        
        # 余弦相似度
        similarity = (user_vec * item_vec).sum(dim=-1) / self.temperature  # [B]
        return similarity, user_vec, item_vec
```

---

## 三、正负样本选择（核心难点）

这是双塔召回效果的决定性因素。工业界有明确共识：

> **选对正负样本的作用 > 改进模型结构** 

### 3.1 正样本

| 定义 | 处理 |
|------|------|
| 曝光且**有效播放**（≥3s）或**完播** | 作为正样本 |
| **热门视频降采样** | 以概率 $p = \min(1, \frac{t}{c_i^{0.75}})$ 丢弃，$c_i$ 为视频点击次数，打压热门 |

### 3.2 负样本：三层体系

短视频场景下，负样本分三层，难度递增：

| 负样本类型 | 来源 | 难度 | 作用 | 采样比例（经验） |
|-----------|------|------|------|----------------|
| **简单负样本（Easy）** | 全局视频库随机采样 | 易区分 | 学习"完全不相关" | 50% |
| **Batch 内负样本** | 同 Batch 其他正样本的视频 | 中等 | 增加负样本多样性，利用 GPU 并行 | 作为补充 |
| **困难负样本（Hard）** | 被召回但被粗排/精排淘汰的视频 | 难区分 | 学习"相关但不够好"的边界 | 50% |

#### 简单负样本采样策略

**绝对不能均匀随机采样！** 二八法则导致正样本大多是热门视频，若负样本均匀采样则多为冷门视频，模型会学偏。

**正确做法：非均匀采样（频率加权）**

$$
P(\text{sample video } i) \propto c_i^{0.75}
$$

其中 $c_i$ 为视频 $i$ 的点击次数。$0.75$ 是经验值（来自 YouTube 论文），目的是**打压热门、提升冷门**。

**训练时纠偏**：根据采样概率修正 logits 

$$
\text{logit}_{\text{corrected}} = \text{cosine}(u, v) - \log P(i)
$$

线上召回时**不纠偏**，用原始余弦相似度。

#### Batch 内负样本（In-batch Negatives）

一个 Batch 有 $N$ 个正样本 $(u_i, v_i)$，对于用户 $u_i$，同 Batch 的其他 $N-1$ 个视频 $v_j (j \neq i)$ 作为负样本。

**优点**：无需额外采样，GPU 并行高效。
**缺点**：热门视频在 Batch 中出现概率高，会被过度打压。需配合纠偏公式。

```python
# Batch 内负样本 + 采样纠偏
def in_batch_negative_loss(user_vec, item_vec, sample_probs, temperature=0.05):
    """
    user_vec: [B, D]
    item_vec: [B, D]
    sample_probs: [B] 每个正样本视频的采样概率
    """
    # 计算所有 user-item 对的相似度矩阵 [B, B]
    logits = torch.matmul(user_vec, item_vec.t()) / temperature  # [B, B]
    
    # 纠偏：减去 log(采样概率)
    logits = logits - torch.log(sample_probs).unsqueeze(0)  # 广播到 [B, B]
    
    # 对角线为正样本
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)  # 每个正样本 vs 所有负样本
    
    return loss
```

#### 困难负样本（Hard Negatives）

来源：
1. **被召回但被粗排淘汰**：说明视频与用户有一定相关性，但不够强
2. **通过粗排但被精排排在尾部**：更难，与正样本非常接近

**为什么不能用"曝光未点击"作为召回负样本？**

> 曝光未点击的视频已经通过了精排的严格筛选，与用户兴趣高度相关。召回的目标是区分"不感兴趣"和"可能感兴趣"，而排序才是区分"比较感兴趣"和"非常感兴趣"。曝光未点击的视频对召回来说甚至**可以算作正样本**。

### 3.3 混合负采样策略（工业界标准）

```python
class NegativeSampler:
    def __init__(self, video_freq, hard_negative_pool, easy_ratio=0.5):
        """
        video_freq: dict {video_id: click_count} 用于频率加权采样
        hard_negative_pool: 被排序淘汰的视频集合
        """
        self.video_freq = video_freq
        self.hard_pool = hard_negative_pool
        self.easy_ratio = easy_ratio
        
        # 预计算采样概率 (频率^0.75)
        self.sampling_probs = {
            vid: (cnt ** 0.75) for vid, cnt in video_freq.items()
        }
        total = sum(self.sampling_probs.values())
        self.sampling_probs = {k: v/total for k, v in self.sampling_probs.items()}
        
    def sample(self, batch_size, user_id=None):
        """为每个正样本采样负样本"""
        n_easy = int(batch_size * self.easy_ratio)
        n_hard = batch_size - n_easy
        
        # 简单负样本：全局频率加权采样
        easy_negs = np.random.choice(
            list(self.sampling_probs.keys()),
            size=n_easy,
            p=list(self.sampling_probs.values()),
            replace=True
        )
        
        # 困难负样本：从该用户被排序淘汰的池中采样
        hard_candidates = self.hard_pool.get(user_id, [])
        if len(hard_candidates) >= n_hard:
            hard_negs = np.random.choice(hard_candidates, size=n_hard, replace=False)
        else:
            # 困难样本不足，补充简单样本
            hard_negs = np.random.choice(
                list(self.sampling_probs.keys()),
                size=n_hard,
                p=list(self.sampling_probs.values())
            )
            
        return list(easy_negs) + list(hard_negs)
```

---

## 四、完整训练流程

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class ShortVideoDataset(Dataset):
    def __init__(self, user_features, item_features, labels, neg_sampler):
        self.user_features = user_features
        self.item_features = item_features
        self.labels = labels
        self.neg_sampler = neg_sampler
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # 正样本
        user_feat = self.user_features[idx]
        pos_item_feat = self.item_features[idx]
        label = self.labels[idx]  # 1.0 for positive
        
        # 采样负样本
        neg_item_ids = self.neg_sampler.sample(1, user_feat['user_id'])
        neg_item_feat = get_item_features(neg_item_ids[0])  # 查特征库
        
        return {
            'user': user_feat,
            'pos_item': pos_item_feat,
            'neg_item': neg_item_feat,
            'label': label
        }

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        user_features = {k: v.to(device) for k, v in batch['user'].items()}
        pos_item_features = {k: v.to(device) for k, v in batch['pos_item'].items()}
        neg_item_features = {k: v.to(device) for k, v in batch['neg_item'].items()}
        
        # 正样本分数
        pos_score, user_vec, pos_item_vec = model(user_features, pos_item_features)
        
        # 负样本分数
        neg_score, _, neg_item_vec = model(user_features, neg_item_features)
        
        # BPR Loss: 鼓励 pos_score > neg_score
        loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
        
        # 或 InfoNCE Loss（配合 Batch 内负样本）
        # loss = in_batch_negative_loss(user_vec, pos_item_vec, sample_probs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)
```

---

## 五、线上部署与更新

| 环节 | 方案 | 说明 |
|------|------|------|
| **Item 向量存储** | Faiss / Milvus (HNSW + PQ) | 千万级视频，64 维向量，内存占用可控 |
| **User 向量计算** | 在线实时推理 | 用户最新行为序列实时更新，秒级延迟 |
| **全量更新** | 每天凌晨用前一天数据训练 1 epoch | 更新模型参数 + 刷新全量 Item 向量 |
| **增量更新** | 每 10-30 分钟更新 ID Embedding | 新发布视频、新注册用户快速生效 |
| **ANN 检索** | HNSW 索引，Top-K=1000 | 余弦相似度，毫秒级响应 |

---

## 六、关键设计要点总结

| 设计点 | 短视频场景的特殊处理 |
|--------|-------------------|
| **Label** | 以有效播放/完播为正样本，而非点击或点赞 |
| **用户塔** | 短期行为序列（Self-Attention）> 长期统计，兴趣漂移快 |
| **视频塔** | 引入作者、BGM、标签等内容特征，新视频冷启友好 |
| **负样本** | **绝不使用曝光未点击**；混合简单（全局频率加权）+ 困难（排序淘汰） |
| **采样纠偏** | 训练时减去 $\log P(i)$，线上不纠偏 |
| **温度系数** | 通常设 0.05~0.1，控制分布锐度 |
| **向量归一化** | User/Item 向量均做 L2 归一化，等价于余弦相似度 |

这套方案是抖音、快手等头部平台的工业界标准实践，核心在于**样本工程 > 模型复杂度**——选对正负样本、做好采样纠偏，比堆叠更复杂的网络结构收益更大。

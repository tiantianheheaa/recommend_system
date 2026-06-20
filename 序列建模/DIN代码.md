- DIN和target-attention的思想是一样的。只是具体实现有一些细微差别。
- target-attention是用候选商品的emb 和 用户行为序列中的emb 做**内积得到scores**，然后softmax的weights。
- DIN是用候选商品emb和用户行为序列emb，通过一个**小型MLP得到scores**。  可以通过softmax归一化、也可以不通过softmax归一化直接用原始mlp输出值。

--- 

# DIN 用户行为序列建模：原理、代码、适用场景与在线耗时

本文按以下结构展开：

1. **DIN 要解决的问题**
2. **DIN 序列建模原理**
3. **特征设计与行为序列截断策略**
4. **完整 PyTorch 代码示例**
5. **在线推理耗时对比**
6. **适用场景与不适用场景**
7. **工业落地优化建议**

---

# 1. DIN 要解决什么问题？

在排序模型中，用户历史行为序列通常包含非常强的兴趣信号，例如：

```text
用户最近点击过：iPhone、手机壳、充电器、无线耳机
候选商品：AirPods
```

如果排序模型只使用用户 ID、商品 ID、类目、品牌等静态特征，模型只能学习到比较粗的用户偏好。

但用户兴趣往往是 **多兴趣、多阶段、强上下文相关** 的：

```text
用户历史行为：
1. 奶粉
2. 纸尿裤
3. 儿童玩具
4. 手机
5. 充电器
6. 手机壳

候选商品 A：婴儿推车
候选商品 B：蓝牙耳机
```

同一个用户，对不同候选商品应该关注不同的历史行为：

| 候选商品 | 应重点关注的历史行为 |
|---|---|
| 婴儿推车 | 奶粉、纸尿裤、儿童玩具 |
| 蓝牙耳机 | 手机、充电器、手机壳 |

**DIN，Deep Interest Network** 的核心思想就是：

> **不是把用户历史行为简单平均，而是针对每个候选商品，动态地从用户历史行为中挑选最相关的行为。**

这就是 DIN 的 **Local Activation Unit，局部激活单元**。

---

# 2. DIN 序列建模原理

## 2.1 传统序列池化的问题

常见的用户行为序列处理方式是：

\[
user\_interest = Pooling(e_1, e_2, ..., e_L)
\]

其中：

- \(e_t\)：第 \(t\) 个历史行为 item 的 embedding
- \(L\)：用户行为序列长度
- Pooling：sum pooling、mean pooling、max pooling

例如：

\[
user\_interest = \frac{1}{L}\sum_{t=1}^{L} e_t
\]

这种方式的问题是：**所有历史行为对当前候选商品的贡献是一样的**。

但真实场景中，用户行为与候选商品之间的相关性不同。

---

## 2.2 DIN 的核心：候选商品触发用户兴趣

DIN 对每个候选商品 \(a\)，都会计算它与用户历史行为 \(e_j\) 的相关性：

\[
w_j = Attention(a, e_j)
\]

然后加权汇聚用户历史行为：

\[
v_U(a) = \sum_{j=1}^{L} w_j e_j
\]

其中：

- \(a\)：候选商品 embedding
- \(e_j\)：用户第 \(j\) 个历史行为 embedding
- \(w_j\)：历史行为 \(j\) 对当前候选商品的兴趣权重
- \(v_U(a)\)：针对候选商品 \(a\) 激活出来的用户兴趣向量

重点是：

> **DIN 得到的用户兴趣向量不是固定的，而是随候选商品变化。**

---

## 2.3 DIN Attention 不是标准 Transformer Attention

DIN 的 attention 通常不是简单点积：

\[
score = q^\top k
\]

而是使用一个小型 MLP 学习候选商品和历史行为之间的非线性关系。

常见输入形式是：

\[
[q, k, q-k, q \odot k]
\]

其中：

- \(q\)：候选商品 embedding
- \(k\)：历史行为 embedding
- \(q-k\)：差异特征
- \(q \odot k\)：逐元素乘积，表示交互特征

然后通过 MLP 得到激活分数：

\[
score_j = MLP([q, k_j, q-k_j, q \odot k_j])
\]

再做 mask，过滤 padding 行为。

可以选择：

### 方式一：softmax attention

\[
w_j = \frac{exp(score_j)}{\sum_{t=1}^{L} exp(score_t)}
\]

### 方式二：DIN 原论文更常见的 activation weight

DIN 中常见做法不一定强制 softmax，而是直接用 activation score 作为权重：

\[
v_U(a)=\sum_{j=1}^{L} score_j e_j
\]

工业实现中两种方式都有：

| 方式 | 特点 |
|---|---|
| **softmax attention** | 权重归一化，训练稳定，解释性更强 |
| **非归一 activation** | 保留兴趣强度，能表达用户整体兴趣强弱 |

---

# 3. 特征设计与行为序列截断策略

## 3.1 排序模型常用特征结构

DIN 排序模型一般输入以下几类特征。

| 特征类型 | 示例 | 是否常用 |
|---|---|---|
| 用户 ID 类特征 | user_id、会员等级、城市、年龄段、性别 | 常用 |
| 候选商品特征 | item_id、sku_id、spu_id、类目、品牌、店铺、价格段 | 常用 |
| 上下文特征 | 场景、入口、时间、设备、网络、地理位置 | 常用 |
| 用户行为序列 | 点击序列、购买序列、加购序列、收藏序列、搜索序列 | 核心 |
| 交叉统计特征 | 用户-类目 CTR、用户-品牌 CTR、商品曝光点击统计 | 常用 |
| 实时特征 | 最近点击、最近搜索、实时购物车、会话行为 | 强烈推荐 |

---

## 3.2 DIN 中常用行为序列

电商排序中，常见序列包括：

| 序列 | 示例 | 作用 |
|---|---|---|
| **点击序列** | 最近点击的商品 | 捕捉短期兴趣 |
| **购买序列** | 最近购买的商品 | 捕捉强偏好和消费能力 |
| **加购序列** | 最近加购商品 | 捕捉高转化意图 |
| **收藏序列** | 收藏商品 | 捕捉中长期兴趣 |
| **搜索词序列** | 最近搜索 query | 捕捉显式需求 |
| **类目序列** | 最近浏览类目 | 缓解 item 稀疏 |
| **品牌序列** | 最近浏览品牌 | 捕捉品牌偏好 |
| **店铺序列** | 最近访问店铺 | 捕捉店铺偏好 |

---

## 3.3 通常截取的用户行为长度

用户行为长度没有固定标准，需要根据业务、延迟、显存、效果综合选择。

常见经验如下：

| 业务场景 | 常用序列长度 | 说明 |
|---|---:|---|
| 电商首页推荐 | **50 ~ 100** | 兼顾兴趣覆盖和在线成本 |
| 商品详情页相关推荐 | **20 ~ 50** | 当前商品强相关，历史不宜太长 |
| 搜索排序 | **20 ~ 50** | query 意图更强，序列作为辅助 |
| 广告排序 | **20 ~ 50** | 延迟要求高，序列通常较短 |
| 内容推荐 | **50 ~ 200** | 用户连续消费行为更密集 |
| 本地生活/酒旅 | **20 ~ 100** | 低频业务可适当拉长时间窗口 |
| 购买序列 | **10 ~ 50** | 购买行为稀疏，长度通常短 |
| 点击序列 | **50 ~ 200** | 点击行为丰富，可以更长 |

电商排序中，一个比较常见的起点是：

```text
click_seq_len = 50 或 100
cart_seq_len = 20 或 50
buy_seq_len = 20 或 50
search_seq_len = 20 或 50
```

如果线上延迟压力较大，可以先从：

```text
click_seq_len = 50
buy_seq_len = 20
cart_seq_len = 20
```

开始实验。

---

# 4. DIN 模型完整 PyTorch 示例

下面代码给出一个相对完整的 DIN 排序模型，包括：

- 特征定义；
- 用户侧稀疏特征；
- 商品侧稀疏特征；
- 上下文特征；
- Dense 连续特征；
- 用户行为序列特征；
- DIN attention 单元；
- DICE 激活函数；
- MLP 排序网络；
- 训练样例。

---

## 4.1 特征定义

```python
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class SparseFeature:
    """
    稀疏离散特征定义。
    name: 特征名
    vocab_size: 词表大小
    embed_dim: embedding 维度
    padding_idx: 是否保留 0 作为 padding
    """
    name: str
    vocab_size: int
    embed_dim: int
    padding_idx: Optional[int] = None


@dataclass
class DenseFeature:
    """
    连续特征定义。
    name: 特征名
    dim: 维度，通常是 1
    """
    name: str
    dim: int = 1


@dataclass
class SequenceFeature:
    """
    用户行为序列特征定义。
    name: 序列名称，例如 hist_click
    fields: 序列中包含的字段，例如 item_id、cate_id、brand_id
    max_len: 最大截断长度
    """
    name: str
    fields: List[SparseFeature]
    max_len: int
```

---

## 4.2 DIN Attention、Dice、MLP

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dice(nn.Module):
    """
    DIN 中常用的 Dice 激活函数。
    Dice 可以看作数据自适应版本的 PReLU。
    """
    def __init__(self, input_dim: int, eps: float = 1e-8):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim, eps=eps)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        """
        支持输入:
        - [B, D]
        - [B, L, D]
        """
        if x.dim() == 2:
            p = torch.sigmoid(self.bn(x))
            return p * x + (1 - p) * self.alpha * x

        elif x.dim() == 3:
            B, L, D = x.shape
            x_reshape = x.reshape(B * L, D)
            p = torch.sigmoid(self.bn(x_reshape)).reshape(B, L, D)
            return p * x + (1 - p) * self.alpha * x

        else:
            raise ValueError("Dice only supports 2D or 3D input.")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int],
        dropout: float = 0.0,
        activation: str = "dice",
        output_dim: Optional[int] = None
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_units:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "dice":
                layers.append(Dice(hidden_dim))
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "prelu":
                layers.append(nn.PReLU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_dim = hidden_dim

        if output_dim is not None:
            layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DINAttention(nn.Module):
    """
    DIN Local Activation Unit。

    query: 候选商品 embedding, shape [B, D]
    keys: 用户历史行为 embedding, shape [B, L, D]
    mask: 有效行为 mask, shape [B, L], True 表示有效
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_units: List[int] = [80, 40],
        activation: str = "dice",
        use_softmax: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_softmax = use_softmax

        # 输入为 [q, k, q-k, q*k]，所以维度是 4 * D
        self.att_mlp = MLP(
            input_dim=4 * embed_dim,
            hidden_units=hidden_units,
            dropout=0.0,
            activation=activation,
            output_dim=1
        )

    def forward(self, query, keys, mask):
        B, L, D = keys.shape

        # [B, D] -> [B, L, D]
        query_expand = query.unsqueeze(1).expand(-1, L, -1)

        att_input = torch.cat(
            [
                query_expand,
                keys,
                query_expand - keys,
                query_expand * keys
            ],
            dim=-1
        )  # [B, L, 4D]

        scores = self.att_mlp(att_input).squeeze(-1)  # [B, L]

        # mask padding 位置
        scores = scores.masked_fill(~mask, -1e9)

        if self.use_softmax:
            weights = F.softmax(scores, dim=-1)  # [B, L]
        else:
            # DIN 原始思想中常见做法：不强制归一化
            weights = torch.where(mask, scores, torch.zeros_like(scores))

        # [B, L] -> [B, L, 1]
        weights = weights.unsqueeze(-1)

        # 加权求和
        user_interest = torch.sum(weights * keys, dim=1)  # [B, D]

        return user_interest, scores
```

---

## 4.3 DIN 主模型

```python
class DINModel(nn.Module):
    def __init__(
        self,
        user_sparse_features: List[SparseFeature],
        item_sparse_features: List[SparseFeature],
        context_sparse_features: List[SparseFeature],
        dense_features: List[DenseFeature],
        behavior_sequence: SequenceFeature,
        mlp_hidden_units: List[int] = [256, 128, 64],
        att_hidden_units: List[int] = [80, 40],
        dropout: float = 0.1,
        use_softmax_attention: bool = False
    ):
        super().__init__()

        self.user_sparse_features = user_sparse_features
        self.item_sparse_features = item_sparse_features
        self.context_sparse_features = context_sparse_features
        self.dense_features = dense_features
        self.behavior_sequence = behavior_sequence

        # 所有稀疏特征统一建 embedding
        all_sparse_features = (
            user_sparse_features
            + item_sparse_features
            + context_sparse_features
            + behavior_sequence.fields
        )

        self.embedding_dict = nn.ModuleDict()
        for feat in all_sparse_features:
            if feat.name not in self.embedding_dict:
                self.embedding_dict[feat.name] = nn.Embedding(
                    num_embeddings=feat.vocab_size,
                    embedding_dim=feat.embed_dim,
                    padding_idx=feat.padding_idx
                )

        # 候选商品 embedding 拼接维度
        self.item_embed_dim = sum([f.embed_dim for f in item_sparse_features])

        # 行为序列 embedding 拼接维度
        self.seq_embed_dim = sum([f.embed_dim for f in behavior_sequence.fields])

        assert self.item_embed_dim == self.seq_embed_dim, (
            "为方便 DIN attention，候选商品 embedding 维度需要与行为序列 embedding 维度一致。"
            f" item_embed_dim={self.item_embed_dim}, seq_embed_dim={self.seq_embed_dim}"
        )

        self.attention = DINAttention(
            embed_dim=self.item_embed_dim,
            hidden_units=att_hidden_units,
            activation="dice",
            use_softmax=use_softmax_attention
        )

        # 计算 DNN 输入维度
        user_dim = sum([f.embed_dim for f in user_sparse_features])
        item_dim = sum([f.embed_dim for f in item_sparse_features])
        context_dim = sum([f.embed_dim for f in context_sparse_features])
        dense_dim = sum([f.dim for f in dense_features])
        din_interest_dim = self.item_embed_dim

        dnn_input_dim = (
            user_dim
            + item_dim
            + context_dim
            + dense_dim
            + din_interest_dim
        )

        self.dnn = MLP(
            input_dim=dnn_input_dim,
            hidden_units=mlp_hidden_units,
            dropout=dropout,
            activation="dice",
            output_dim=1
        )

    def embed_sparse_features(self, x: Dict[str, torch.Tensor], features: List[SparseFeature]):
        emb_list = []
        for feat in features:
            emb = self.embedding_dict[feat.name](x[feat.name].long())
            emb_list.append(emb)
        return torch.cat(emb_list, dim=-1)

    def embed_sequence_features(self, x: Dict[str, torch.Tensor]):
        """
        对行为序列中的多个字段做 embedding 后拼接。
        输入 shape:
        x["hist_item_id"]: [B, L]
        x["hist_cate_id"]: [B, L]
        x["hist_brand_id"]: [B, L]

        输出:
        seq_emb: [B, L, D]
        """
        emb_list = []
        for feat in self.behavior_sequence.fields:
            field_name = f"{self.behavior_sequence.name}_{feat.name}"
            emb = self.embedding_dict[feat.name](x[field_name].long())
            emb_list.append(emb)

        seq_emb = torch.cat(emb_list, dim=-1)
        return seq_emb

    def forward(self, x: Dict[str, torch.Tensor]):
        # 用户特征 embedding
        user_emb = self.embed_sparse_features(x, self.user_sparse_features)

        # 候选商品 embedding
        item_emb = self.embed_sparse_features(x, self.item_sparse_features)

        # 上下文 embedding
        if len(self.context_sparse_features) > 0:
            context_emb = self.embed_sparse_features(x, self.context_sparse_features)
        else:
            context_emb = None

        # dense 特征
        dense_list = []
        for feat in self.dense_features:
            dense_list.append(x[feat.name].float())

        if len(dense_list) > 0:
            dense_values = torch.cat(dense_list, dim=-1)
        else:
            dense_values = None

        # 行为序列 embedding
        seq_emb = self.embed_sequence_features(x)

        # mask: seq_len 内为 True
        B, L, _ = seq_emb.shape
        seq_len = x[f"{self.behavior_sequence.name}_seq_len"].long()
        position = torch.arange(L, device=seq_emb.device).unsqueeze(0).expand(B, L)
        mask = position < seq_len.unsqueeze(1)

        # DIN attention
        din_interest, att_scores = self.attention(
            query=item_emb,
            keys=seq_emb,
            mask=mask
        )

        # 拼接 DNN 输入
        dnn_inputs = [user_emb, item_emb, din_interest]

        if context_emb is not None:
            dnn_inputs.append(context_emb)

        if dense_values is not None:
            dnn_inputs.append(dense_values)

        dnn_input = torch.cat(dnn_inputs, dim=-1)

        logit = self.dnn(dnn_input).squeeze(-1)
        prob = torch.sigmoid(logit)

        return {
            "logit": logit,
            "prob": prob,
            "att_scores": att_scores,
            "din_interest": din_interest
        }
```

---

## 4.4 示例特征配置

下面给出一个电商 CTR 排序场景的特征定义。

```python
# 用户稀疏特征
user_sparse_features = [
    SparseFeature("user_id", vocab_size=50_000_000, embed_dim=16),
    SparseFeature("user_level", vocab_size=20, embed_dim=4),
    SparseFeature("user_city", vocab_size=5000, embed_dim=8),
    SparseFeature("user_gender", vocab_size=4, embed_dim=4),
    SparseFeature("user_age_bucket", vocab_size=20, embed_dim=4),
]

# 候选商品稀疏特征
item_sparse_features = [
    SparseFeature("item_id", vocab_size=100_000_000, embed_dim=32),
    SparseFeature("cate_id", vocab_size=20_000, embed_dim=16),
    SparseFeature("brand_id", vocab_size=5_000_000, embed_dim=16),
]

# 上下文特征
context_sparse_features = [
    SparseFeature("scene_id", vocab_size=100, embed_dim=4),
    SparseFeature("device_type", vocab_size=10, embed_dim=4),
    SparseFeature("hour", vocab_size=24, embed_dim=4),
    SparseFeature("weekday", vocab_size=7, embed_dim=4),
]

# 连续特征
dense_features = [
    DenseFeature("item_price_norm", dim=1),
    DenseFeature("item_ctr_7d", dim=1),
    DenseFeature("item_cvr_7d", dim=1),
    DenseFeature("user_ctr_7d", dim=1),
    DenseFeature("user_order_cnt_30d", dim=1),
    DenseFeature("user_item_cate_ctr_30d", dim=1),
]

# 行为序列字段
# 注意：这里复用 item_id、cate_id、brand_id 的 embedding
behavior_sequence = SequenceFeature(
    name="hist_click",
    fields=[
        SparseFeature("item_id", vocab_size=100_000_000, embed_dim=32, padding_idx=0),
        SparseFeature("cate_id", vocab_size=20_000, embed_dim=16, padding_idx=0),
        SparseFeature("brand_id", vocab_size=5_000_000, embed_dim=16, padding_idx=0),
    ],
    max_len=50
)

model = DINModel(
    user_sparse_features=user_sparse_features,
    item_sparse_features=item_sparse_features,
    context_sparse_features=context_sparse_features,
    dense_features=dense_features,
    behavior_sequence=behavior_sequence,
    mlp_hidden_units=[256, 128, 64],
    att_hidden_units=[80, 40],
    dropout=0.1,
    use_softmax_attention=False
)
```

---

## 4.5 构造一个 Batch 输入

```python
B = 4
L = behavior_sequence.max_len

batch = {
    # 用户特征
    "user_id": torch.randint(1, 50_000_000, (B,)),
    "user_level": torch.randint(1, 20, (B,)),
    "user_city": torch.randint(1, 5000, (B,)),
    "user_gender": torch.randint(1, 4, (B,)),
    "user_age_bucket": torch.randint(1, 20, (B,)),

    # 候选商品特征
    "item_id": torch.randint(1, 100_000_000, (B,)),
    "cate_id": torch.randint(1, 20_000, (B,)),
    "brand_id": torch.randint(1, 5_000_000, (B,)),

    # 上下文特征
    "scene_id": torch.randint(1, 100, (B,)),
    "device_type": torch.randint(1, 10, (B,)),
    "hour": torch.randint(0, 24, (B,)),
    "weekday": torch.randint(0, 7, (B,)),

    # 连续特征，shape 必须是 [B, 1]
    "item_price_norm": torch.randn(B, 1),
    "item_ctr_7d": torch.rand(B, 1),
    "item_cvr_7d": torch.rand(B, 1),
    "user_ctr_7d": torch.rand(B, 1),
    "user_order_cnt_30d": torch.rand(B, 1),
    "user_item_cate_ctr_30d": torch.rand(B, 1),

    # 行为序列，shape [B, L]
    "hist_click_item_id": torch.randint(1, 100_000_000, (B, L)),
    "hist_click_cate_id": torch.randint(1, 20_000, (B, L)),
    "hist_click_brand_id": torch.randint(1, 5_000_000, (B, L)),

    # 每个用户真实行为长度
    "hist_click_seq_len": torch.tensor([50, 35, 10, 0]),
}

# 对 seq_len = 0 的用户，将序列置为 padding
for i in range(B):
    valid_len = batch["hist_click_seq_len"][i].item()
    if valid_len < L:
        batch["hist_click_item_id"][i, valid_len:] = 0
        batch["hist_click_cate_id"][i, valid_len:] = 0
        batch["hist_click_brand_id"][i, valid_len:] = 0

out = model(batch)

print(out["prob"])
print(out["att_scores"].shape)
```

---

## 4.6 训练代码示例

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
criterion = nn.BCEWithLogitsLoss()

labels = torch.tensor([1, 0, 1, 0]).float()

model.train()
optimizer.zero_grad()

out = model(batch)
loss = criterion(out["logit"], labels)

loss.backward()
optimizer.step()

print("loss:", loss.item())
```

---

## 4.7 多行为序列扩展示例

实际工业系统中，通常不只用点击序列，还会加入：

```text
点击序列 click_seq
加购序列 cart_seq
购买序列 buy_seq
收藏序列 fav_seq
搜索词序列 search_seq
```

可以对每种行为序列各自做 DIN attention：

\[
v_{click}=DIN(q, click\_seq)
\]

\[
v_{cart}=DIN(q, cart\_seq)
\]

\[
v_{buy}=DIN(q, buy\_seq)
\]

然后拼接：

\[
v=[v_{click}, v_{cart}, v_{buy}]
\]

工程上常见做法：

| 行为类型 | 序列长度 | 权重建议 |
|---|---:|---|
| 点击 | 50 ~ 100 | 主序列 |
| 加购 | 20 ~ 50 | 高转化意图 |
| 购买 | 20 ~ 50 | 强偏好 |
| 收藏 | 20 ~ 50 | 中长期兴趣 |
| 搜索 | 10 ~ 50 | 显式需求 |

---

# 5. 在线耗时对比：有序列建模 vs 无序列建模

## 5.1 复杂度对比

假设：

- \(B\)：一次请求排序候选数，例如 200、500、1000
- \(L\)：用户行为序列长度，例如 50
- \(D\)：embedding 维度，例如 64
- \(H\)：attention MLP 隐层维度

### 无序列建模

模型大致是：

```text
用户特征 embedding + 商品特征 embedding + 上下文特征 + MLP
```

复杂度主要来自：

\[
O(B \cdot MLP)
\]

### DIN 序列建模

DIN 需要对每个候选商品和用户历史行为逐一计算 attention：

\[
O(B \cdot L \cdot AttentionMLP)
\]

也就是：

```text
每个候选商品 × 每条用户历史行为 × 一个 attention 小网络
```

所以 DIN 的在线成本明显更高。

---

## 5.2 在线耗时经验范围

不同公司、硬件、候选量、模型结构、特征服务、Batch 策略差异很大，不能给绝对固定值。下面是工业排序服务中常见的 **经验级参考**。

假设：

```text
排序候选数：200 ~ 500
Embedding 维度：32 ~ 128
序列长度：50
推理方式：CPU 或 GPU Batch 推理
模型：3 层 MLP + DIN attention
```

| 模型 | 典型在线耗时 | 相对耗时 | 说明 |
|---|---:|---:|---|
| **无序列 DNN 排序模型** | 5 ~ 20 ms | 1.0x | 只用静态用户、商品、上下文特征 |
| **Mean Pooling 序列模型** | 6 ~ 25 ms | 1.1x ~ 1.5x | 序列只聚合一次，对候选不敏感 |
| **DIN，L=20** | 10 ~ 35 ms | 1.5x ~ 2.5x | 延迟可控，适合广告/搜索 |
| **DIN，L=50** | 15 ~ 60 ms | 2x ~ 4x | 电商推荐常见配置 |
| **DIN，L=100** | 30 ~ 120 ms | 3x ~ 6x | 需要较强优化或 GPU 推理 |
| **多序列 DIN** | 40 ~ 150 ms+ | 4x ~ 8x+ | 点击、加购、购买多序列叠加 |

更细一点，从单候选角度看：

| 模型 | 单候选相对成本 |
|---|---:|
| 无序列 MLP | 1 |
| Mean Pooling + MLP | 1.1 ~ 1.3 |
| DIN L=20 | 1.5 ~ 2.5 |
| DIN L=50 | 2.5 ~ 4.5 |
| DIN L=100 | 4 ~ 8 |

核心结论：

> **DIN 的主要在线开销来自“候选商品 × 行为序列长度”的交叉 attention。**

---

## 5.3 为什么 DIN 比普通 DNN 慢？

普通 DNN 对每个候选商品只计算一次：

```text
score(user, item)
```

DIN 对每个候选商品，需要额外计算：

```text
score(item, hist_item_1)
score(item, hist_item_2)
...
score(item, hist_item_L)
```

如果候选数是 500，序列长度是 50，则每个请求需要：

\[
500 \times 50 = 25000
\]

次候选商品与历史行为的交互计算。

这就是 DIN 在线成本高的根本原因。

---

# 6. DIN 的适用场景

## 6.1 非常适合 DIN 的场景

### 1. 用户兴趣多样化明显

例如电商用户可能同时关注：

```text
母婴、手机、美妆、食品、家电
```

DIN 可以针对不同候选商品激活不同兴趣。

---

### 2. 候选商品与历史行为强相关

例如：

| 用户历史行为 | 当前候选商品 |
|---|---|
| 手机、手机壳、充电器 | 无线耳机 |
| 奶粉、纸尿裤、童装 | 婴儿推车 |
| 跑鞋、运动裤、健身器材 | 蛋白粉 |

这种场景 DIN 往往能明显提升 CTR/CVR。

---

### 3. 排序阶段候选量有限

DIN 不适合对全库商品直接打分，但适合用于：

```text
召回后 200 ~ 1000 个候选商品的排序
```

候选量越小，DIN 的在线成本越容易接受。

---

### 4. 用户行为数据丰富

DIN 依赖历史行为，如果用户行为稀疏，效果不一定明显。

适合：

- 高频电商首页；
- 信息流推荐；
- 短视频推荐；
- 广告点击率预估；
- 商品详情页推荐；
- 内容 feed 排序。

---

## 6.2 不太适合 DIN 的场景

| 场景 | 原因 |
|---|---|
| 极低延迟广告请求 | DIN attention 成本较高 |
| 用户行为极少 | 序列信息不足 |
| 候选数极大 | \(B \cdot L\) 成本过高 |
| 强 query 主导搜索 | query 意图可能比历史行为更重要 |
| 大量新用户冷启动 | 历史序列为空 |
| 需要极强长期序列依赖 | DIN 不建模行为顺序，可能不如 Transformer |

---

# 7. DIN 的优缺点

## 7.1 优点

| 优点 | 说明 |
|---|---|
| **候选商品自适应兴趣激活** | 同一个用户面对不同商品时，激活不同历史兴趣 |
| **比 mean pooling 更精细** | 不会把多个兴趣简单平均 |
| **结构相对简单** | 比 Transformer 轻，工业落地成熟 |
| **可解释性较好** | attention score 可用于分析哪些历史行为影响当前预测 |
| **适合排序模型** | 在召回后的小候选集上效果较好 |

---

## 7.2 缺点

| 缺点 | 说明 |
|---|---|
| **在线耗时高于普通 DNN** | 复杂度随候选数和序列长度线性增长 |
| **不建模行为顺序** | DIN 关注相关性，不强调时序演化 |
| **长序列成本高** | L 从 50 增加到 100，attention 成本近似翻倍 |
| **对特征服务要求高** | 需要在线实时读取用户行为序列 |
| **冷启动用户收益有限** | 没有历史行为时退化为普通 DNN |
| **多序列扩展成本明显增加** | 点击、购买、加购多序列会显著增加推理压力 |

---

# 8. 工业落地优化建议

## 8.1 序列截断策略

不要简单取最近 N 个行为，可以综合使用：

### 1. 按时间截断

```text
取最近 50 个点击行为
```

优点是简单、实时性强。

### 2. 按行为类型分桶

```text
点击 50
加购 20
购买 20
收藏 20
```

不同信号分开建模，效果通常更好。

### 3. 按类目去重

避免用户近期大量点击同一类商品导致序列冗余。

```text
同一 item 去重
同一店铺限频
同一类目保留最近若干个
```

### 4. 时间窗口限制

例如：

```text
点击：最近 30 天
加购：最近 90 天
购买：最近 180 天
```

### 5. 行为权重衰减

越近的行为权重越高：

\[
weight = exp(-\lambda \Delta t)
\]

可以作为额外特征输入模型。

---

## 8.2 在线性能优化

### 1. 控制候选数量

DIN 更适合精排或粗排后段：

```text
召回 5000
粗排 1000
DIN 精排 200 ~ 500
```

如果直接对 5000 个候选做 DIN，成本通常较高。

---

### 2. 控制序列长度

建议从以下配置起步：

```text
click_seq_len = 50
cart_seq_len = 20
buy_seq_len = 20
```

再根据线上收益逐步增加。

---

### 3. 序列 embedding 预取

用户历史行为序列的 ID 可以在线存储，但 embedding lookup 尽量批量化。

常见链路：

```text
用户行为 KV 服务
        ↓
批量 ID 特征
        ↓
Embedding Service / 本地 embedding cache
        ↓
DIN 推理服务
```

---

### 4. 共享历史序列计算

用户历史序列对同一个请求内所有候选是相同的，可以复用：

```text
hist_item_embedding
hist_cate_embedding
hist_brand_embedding
```

但 attention 仍然需要针对每个候选商品计算。

---

### 5. 使用更轻的 attention 网络

例如将 attention MLP 从：

```text
[80, 40]
```

缩小为：

```text
[64, 32]
[32, 16]
```

或者将 embedding 维度从 128 降到 64。

---

### 6. 使用粗排过滤

链路建议：

```text
召回 TopN = 5000
轻量粗排 TopM = 500 ~ 1000
DIN 精排 TopK = 100 ~ 300
重排
```

DIN 不建议直接放在过大的候选集上。

---

### 7. TensorRT / ONNX / GPU Batch 推理

如果线上 QPS 高、候选数较大，可以使用：

- ONNX Runtime；
- TensorRT；
- GPU Batch 推理；
- FP16；
- INT8 量化；
- embedding cache；
- attention kernel fusion。

---

# 9. DIN 与其他序列建模方法对比

| 方法 | 是否关注候选商品 | 是否建模顺序 | 在线成本 | 适用阶段 |
|---|---:|---:|---:|---|
| **Mean Pooling** | 否 | 否 | 低 | 粗排/精排 |
| **DIN** | 是 | 否 | 中高 | 精排 |
| **DIEN** | 是 | 是 | 高 | 精排 |
| **DSIN** | 是 | 是，按 session | 高 | 精排 |
| **BST/Transformer** | 可支持 | 是 | 高 | 精排/重排 |
| **SIM** | 是 | 部分支持 | 中高 | 长序列精排 |
| **MIMN** | 间接支持 | 长期兴趣记忆 | 中高 | 高活用户场景 |

DIN 的定位可以概括为：

> **比简单 pooling 更强，比 Transformer 更轻，是工业排序中非常经典的候选商品感知型序列建模方法。**

---

# 10. 实战推荐配置

如果是在电商排序场景中首次上线 DIN，可以从如下配置开始：

```text
候选数：200 ~ 500
点击序列长度：50
加购序列长度：20
购买序列长度：20
embedding 维度：
    item_id: 32
    cate_id: 16
    brand_id: 16
attention MLP:
    [80, 40] 或 [64, 32]
主 MLP:
    [256, 128, 64]
激活函数：
    Dice 或 PReLU
loss:
    BCEWithLogitsLoss
优化器：
    Adam / Adagrad
```

如果线上延迟超标，优先优化顺序建议：

1. **减少精排候选数**
2. **缩短行为序列长度**
3. **降低 embedding 维度**
4. **缩小 attention MLP**
5. **减少序列类型**
6. **使用 GPU / TensorRT / FP16**
7. **引入两阶段排序：轻 DIN 粗排 + 重模型精排**

---

# 核心结论

1. **DIN 的核心是候选商品触发用户兴趣**，不是对历史行为做简单平均。
2. **DIN attention 会针对每个候选商品重新计算用户历史行为权重**，因此表达能力强，但在线成本更高。
3. 电商排序中，点击序列常用长度一般是 **50 ~ 100**，购买、加购序列一般是 **20 ~ 50**。
4. 在线耗时方面，DIN 通常比无序列 DNN 慢 **2 ~ 4 倍**，长序列或多序列场景可能达到 **4 ~ 8 倍以上**。
5. DIN 适合放在 **召回之后、候选数有限的排序阶段**，不适合直接对全量商品打分。
6. 工业落地时，应重点控制 **候选数、序列长度、embedding 维度、attention MLP 复杂度**。

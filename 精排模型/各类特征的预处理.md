- attention在推荐系统和LLM中的应用。
- attention的计算过程（1）乘法：计算相似度score (2)softmax归一化：得到权重weight。（3）乘法：加权求和。
- attention在推荐系统 序列建模中的target-attention。 候选item是一个元素，计算和整个序列的每个元素的attention score。 **weight是一维数组**。
- attention在LLM中，对整个句子进行self-attention。 需要计算句子中每个元素 对 句子中每个位置的权重。因此**weight是二维数组**。
- multi head attention，就是把emb_dim分为了多头。即把[batch_size, seq_len, emb_dim] 分隔为了[batch_size, seq_len, num_head, head_dim]


---

# 深度推荐精排模型中的特征预处理与输入方式

在推荐系统精排模型中，输入特征通常可以分为几类：

1. **Dense 特征**：连续数值特征，例如价格、年龄、点击率、曝光次数、停留时长。
2. **Sparse 特征**：高维稀疏 ID 类特征，例如用户 ID、商品 ID、店铺 ID、品牌 ID、类目 ID。
3. **Category 特征**：离散类别特征，例如性别、城市、会员等级、商品品类、设备类型。广义上属于 sparse 特征的一种。
4. **用户行为序列特征**：用户最近点击、购买、加购、搜索过的商品或类目序列，例如最近 50 个点击商品 ID。

这些特征的核心目标是：**把原始业务数据转换成神经网络可处理的数值张量**。

---

# 1. 总体流程

一个典型推荐精排模型的输入处理流程如下：

```text
原始样本
  ↓
特征抽取
  ↓
特征清洗与缺失值处理
  ↓
特征编码
  ↓
数值归一化 / 离散化 / ID 映射
  ↓
Embedding 查表 / Dense 拼接 / Sequence 编码
  ↓
特征融合
  ↓
输入 DNN / Attention / DIN / Transformer / MMOE 等模型
  ↓
输出 CTR / CVR / GMV / 多目标分数
```

以精排模型为例，最终输入模型的通常不是原始字段，而是下面几类张量：

```text
dense_tensor:      [batch_size, dense_dim]
sparse_embeddings: [batch_size, num_sparse_fields, embed_dim]
seq_embeddings:    [batch_size, seq_len, embed_dim]
mask_tensor:       [batch_size, seq_len]
```

---

# 2. Dense 特征：连续数值特征

## 2.1 Dense 特征是什么？

Dense 特征是连续型或数值型特征，通常每个样本都有明确数值。

常见例子：

| 特征 | 示例 | 含义 |
|---|---:|---|
| `user_age` | 28 | 用户年龄 |
| `item_price` | 199.0 | 商品价格 |
| `user_click_7d` | 35 | 用户近 7 天点击次数 |
| `item_ctr_1d` | 0.032 | 商品近 1 天点击率 |
| `shop_score` | 4.8 | 店铺评分 |
| `user_avg_dwell_time` | 12.5 | 用户平均停留时长 |
| `item_sales_30d` | 12000 | 商品近 30 天销量 |

---

## 2.2 Dense 特征为什么需要预处理？

神经网络对输入数值尺度比较敏感。比如：

```text
年龄：20 ~ 60
价格：1 ~ 100000
CTR：0 ~ 1
销量：0 ~ 1000000
```

如果直接输入，数值大的特征可能主导梯度，导致模型训练不稳定。

因此 dense 特征通常需要做：

1. **缺失值处理**
2. **异常值截断**
3. **归一化 / 标准化**
4. **Log 变换**
5. **分桶离散化**
6. **统计特征平滑**

---

## 2.3 缺失值处理

常见方式：

| 方法 | 示例 | 适用场景 |
|---|---|---|
| 填 0 | 缺失价格填 0 | 0 有明确业务含义时 |
| 填均值 / 中位数 | 年龄缺失填平均年龄 | 稳定连续特征 |
| 填特殊值 | 缺失填 -1 | 让模型识别缺失状态 |
| 增加 missing indicator | `is_price_missing=1` | 缺失本身有信息量 |

例如：

```text
原始：
item_price = null

处理：
item_price = 0
is_item_price_missing = 1
```

这样模型不仅知道价格值，还知道这个值是补出来的。

---

## 2.4 异常值截断

推荐系统中很多统计特征是长尾分布，例如销量、点击数、曝光数。

```text
大部分商品销量 < 1000
少数爆品销量 > 1000000
```

如果不处理，极端值会影响训练。

常见处理：

```text
item_sales_30d = min(item_sales_30d, P99)
```

例如：

```text
P99 = 50000

原始销量：
item_sales_30d = 1200000

截断后：
item_sales_30d = 50000
```

也可以设置上下界：

```text
x = max(min(x, upper_bound), lower_bound)
```

---

## 2.5 Log 变换

对曝光、点击、销量、消费金额等长尾特征，常用：

```text
x' = log(1 + x)
```

例如：

| 原始销量 | log(1+x) |
|---:|---:|
| 0 | 0 |
| 10 | 2.397 |
| 1000 | 6.909 |
| 1000000 | 13.816 |

Log 变换可以压缩极端大值，使分布更平滑。

---

## 2.6 标准化与归一化

### 标准化

```text
x' = (x - mean) / std
```

适合近似正态分布的特征，例如：

```text
user_age
shop_score
user_avg_dwell_time
```

### Min-Max 归一化

```text
x' = (x - min) / (max - min)
```

将特征压缩到 `[0, 1]`。

适合有明确上下界的特征，例如：

```text
discount_rate
ctr
score
```

### 示例

```text
item_price = 199
mean_price = 100
std_price = 50

标准化后：
item_price_norm = (199 - 100) / 50 = 1.98
```

---

## 2.7 Dense 特征如何输入神经网络？

处理后的 dense 特征通常直接拼成一个向量：

```text
dense_features = [
  user_age_norm,
  item_price_norm,
  item_ctr_norm,
  user_click_7d_norm,
  shop_score_norm
]
```

假设有 20 个 dense 特征，batch size 为 1024，则输入张量为：

```text
dense_tensor.shape = [1024, 20]
```

然后可以直接输入 DNN，也可以先过一层全连接变换：

```text
dense_hidden = Dense(dense_tensor)
```

例如：

```python
dense_input = [
    user_age_norm,
    item_price_norm,
    item_ctr_norm,
    user_click_7d_norm,
    shop_score_norm
]

# dense_input shape: [batch_size, dense_dim]
```

---

# 3. Sparse 特征：高维稀疏 ID 特征

## 3.1 Sparse 特征是什么？

Sparse 特征通常是离散 ID，取值空间巨大，但每条样本只命中少量取值。

常见例子：

| 特征 | 示例 | 取值空间 |
|---|---|---:|
| `user_id` | 123456 | 亿级 |
| `sku_id` | 10099320302155 | 亿级 |
| `shop_id` | 8888 | 千万级 |
| `brand_id` | 123 | 百万级 |
| `category_id` | 670 | 万级 |
| `city_id` | 1 | 千级 |
| `device_id` | iPhone15 | 万级 |

这些特征不能直接作为数值输入神经网络。

例如：

```text
sku_id = 10099320302155
```

这个数字本身没有连续数值含义。`sku_id=100` 不代表比 `sku_id=99` 大，也不代表距离更近。

所以 sparse 特征通常要做：

1. **词表映射**
2. **Hash 编码**
3. **Embedding 查表**
4. **多值特征 Pooling**
5. **低频过滤与 OOV 处理**

---

## 3.2 One-hot 编码为什么不够？

对于一个商品 ID 特征，如果有 1 亿个商品，则 one-hot 向量维度是：

```text
[0, 0, 0, ..., 1, ..., 0]
```

维度可能达到上亿，非常稀疏，无法直接送入 DNN。

Embedding 的作用是把高维稀疏 ID 映射成低维稠密向量。

```text
sku_id = 10099320302155
        ↓
embedding lookup
        ↓
sku_embedding = [0.12, -0.08, 0.33, ..., 0.01]
```

如果 embedding 维度是 64，则：

```text
sku_embedding.shape = [64]
```

---

## 3.3 Sparse 特征预处理方式

### 方式一：词表映射 Vocabulary

维护一个 ID 到 index 的映射表。

```text
sku_id            index
10099320302155 -> 1
10099320302156 -> 2
10099320302157 -> 3
unknown         -> 0
```

输入模型时使用 index：

```text
sku_id = 10099320302155
sku_index = 1
```

再通过 embedding table 查向量：

```text
sku_embedding = embedding_table[1]
```

适合高频、稳定、重要的 ID 特征，例如：

```text
user_id
sku_id
category_id
brand_id
shop_id
```

优点：

- 可控性强
- 支持低频过滤
- 可与预训练 embedding 对齐

缺点：

- 需要维护词表
- 新 ID 需要 OOV 或增量更新
- 大规模 ID 时存储成本高

---

### 方式二：Hash Trick

将原始 ID 通过 hash 映射到固定桶数。

```text
index = hash(sku_id) % bucket_size
```

例如：

```text
bucket_size = 10000000
sku_id = 10099320302155
index = hash(sku_id) % 10000000
```

优点：

- 不需要维护完整词表
- 天然支持新 ID
- 简化在线服务

缺点：

- 存在 hash 冲突
- 不同 ID 可能共享 embedding
- 对超高价值 ID 可能不够精细

---

## 3.4 Embedding 表

每个 sparse field 通常有自己的 embedding table。

例如：

| 特征 | 词表大小 | Embedding 维度 |
|---|---:|---:|
| `user_id` | 1 亿 | 64 |
| `sku_id` | 5 亿 | 64 |
| `brand_id` | 100 万 | 16 |
| `category_id` | 10 万 | 16 |
| `shop_id` | 1000 万 | 32 |
| `city_id` | 1000 | 8 |
| `device_type` | 100 | 8 |

Embedding table 形状：

```text
user_embedding_table: [num_users, 64]
sku_embedding_table:  [num_skus, 64]
brand_embedding_table:[num_brands, 16]
```

查表后：

```text
user_emb = user_embedding_table[user_index]
sku_emb = sku_embedding_table[sku_index]
brand_emb = brand_embedding_table[brand_index]
```

---

## 3.5 Sparse 特征如何输入神经网络？

通常流程：

```text
原始 ID
  ↓
映射成 index
  ↓
embedding lookup
  ↓
得到低维 dense embedding
  ↓
与其他 embedding / dense 特征拼接
  ↓
输入 DNN
```

例如一条样本：

```json
{
  "user_id": "u123",
  "sku_id": "sku456",
  "brand_id": "b10",
  "category_id": "c88",
  "city_id": "beijing"
}
```

映射后：

```json
{
  "user_id": 12345,
  "sku_id": 98765,
  "brand_id": 10,
  "category_id": 88,
  "city_id": 1
}
```

查 embedding 后：

```text
user_emb:     [64]
sku_emb:      [64]
brand_emb:    [16]
category_emb: [16]
city_emb:     [8]
```

拼接：

```text
sparse_concat = concat([
  user_emb,
  sku_emb,
  brand_emb,
  category_emb,
  city_emb
])
```

最终：

```text
sparse_concat.shape = [64 + 64 + 16 + 16 + 8] = [168]
```

再与 dense 特征拼接：

```text
model_input = concat([sparse_concat, dense_features])
```

---

# 4. Category 特征：类别特征

## 4.1 Category 特征与 Sparse 特征的关系

Category 特征本质上也是离散特征，通常可以看作 sparse 特征的一种。

区别在于：

| 类型 | 特点 | 示例 |
|---|---|---|
| Sparse ID 特征 | 取值空间巨大，ID 无明显语义 | `user_id`, `sku_id` |
| Category 特征 | 类别数量较小或中等，语义明确 | `gender`, `city`, `device_type`, `member_level` |

例如：

```text
gender = male / female / unknown
device_type = iOS / Android / PC / MiniProgram
member_level = new / normal / vip / plus
```

---

## 4.2 Category 特征预处理方式

### 方式一：Label Encoding + Embedding

将类别映射成整数 ID：

```text
gender:
unknown -> 0
male    -> 1
female  -> 2
```

然后查 embedding：

```text
gender_emb = embedding_table[gender_index]
```

适合类别数较多，或者希望模型学习类别之间隐式关系的场景。

---

### 方式二：One-hot 编码

类别数量非常少时，可以直接 one-hot。

例如：

```text
device_type = Android
```

编码为：

```text
iOS      Android      PC      MiniProgram
0        1            0       0
```

适合：

```text
gender
是否会员
是否新用户
是否促销
```

缺点是类别多时维度变大。

---

### 方式三：Multi-hot 编码

如果一个特征可以同时有多个类别，例如商品标签：

```text
item_tags = ["低价", "新品", "自营"]
```

可编码成 multi-hot：

```text
低价 新品 自营 包邮 品牌
1    1    1    0    0
```

在深度模型中，更常见的是：

```text
item_tags
  ↓
tag_id list
  ↓
embedding lookup
  ↓
mean/sum/max pooling
```

例如：

```text
tag_embeddings = [
  emb("低价"),
  emb("新品"),
  emb("自营")
]

item_tag_emb = mean_pooling(tag_embeddings)
```

---

## 4.3 Category 特征输入方式示例

假设有以下类别特征：

```json
{
  "gender": "male",
  "city": "beijing",
  "device": "ios",
  "member_level": "plus"
}
```

编码：

```json
{
  "gender": 1,
  "city": 10,
  "device": 2,
  "member_level": 4
}
```

Embedding：

```text
gender_emb:       [4]
city_emb:         [8]
device_emb:       [4]
member_level_emb: [4]
```

拼接：

```text
category_emb = concat([
  gender_emb,
  city_emb,
  device_emb,
  member_level_emb
])
```

---

# 5. 用户行为序列特征

## 5.1 用户行为序列是什么？

用户行为序列特征描述用户历史兴趣，是推荐系统精排模型中非常重要的一类特征。

常见序列：

| 序列类型 | 示例 |
|---|---|
| 最近点击商品序列 | `click_sku_seq = [sku1, sku2, sku3, ...]` |
| 最近购买商品序列 | `buy_sku_seq = [sku7, sku8, ...]` |
| 最近加购商品序列 | `cart_sku_seq = [sku4, sku5, ...]` |
| 最近搜索词序列 | `query_seq = ["手机", "耳机", "充电器"]` |
| 最近浏览类目序列 | `category_seq = [c1, c3, c8]` |
| 长期兴趣序列 | 用户半年内高频互动类目 |
| 短期兴趣序列 | 用户最近 30 分钟点击商品 |

行为序列体现用户兴趣的动态变化，例如：

```text
用户最近点击：
[手机壳, iPhone15, 无线充电器, 蓝牙耳机]

候选商品：
AirPods

模型可以判断：用户当前可能对手机配件或数码产品感兴趣。
```

---

## 5.2 行为序列预处理

### 5.2.1 按时间排序

通常按照行为发生时间排序：

```text
从早到晚：
[sku1, sku2, sku3, sku4]

从近到远：
[sku4, sku3, sku2, sku1]
```

不同模型可能使用不同顺序。

- RNN/Transformer 通常使用时间顺序。
- DIN/Attention 类模型通常关心候选 item 与历史 item 的相关性，对顺序要求相对弱一些。
- DIEN/Transformer 更依赖行为演化顺序。

---

### 5.2.2 截断

用户历史行为可能很长，不可能全部输入模型。

例如：

```text
用户过去一年点击了 10000 个商品
```

实际模型可能只保留最近 50 或 100 个：

```text
click_sku_seq = last_50_clicks
```

常见长度：

| 序列 | 常用最大长度 |
|---|---:|
| 点击商品序列 | 50 / 100 / 200 |
| 购买商品序列 | 20 / 50 |
| 加购商品序列 | 20 / 50 |
| 搜索词序列 | 10 / 20 |
| 类目序列 | 50 / 100 |

截断策略：

```text
保留最近 N 个
保留高价值行为
按时间窗口保留
按行为权重采样
```

---

### 5.2.3 Padding

不同用户行为长度不同：

```text
用户 A: [sku1, sku2, sku3]
用户 B: [sku4, sku5, sku6, sku7, sku8]
```

为了组成 batch，需要 padding 到固定长度。

假设最大长度为 5：

```text
用户 A: [sku1, sku2, sku3, PAD, PAD]
用户 B: [sku4, sku5, sku6, sku7, sku8]
```

同时生成 mask：

```text
用户 A mask: [1, 1, 1, 0, 0]
用户 B mask: [1, 1, 1, 1, 1]
```

mask 的作用是告诉模型：

```text
1 表示真实行为
0 表示 padding，不参与计算
```

---

### 5.2.4 去重与保留重复

是否去重取决于业务和模型设计。

#### 不去重

```text
[sku1, sku2, sku1, sku3]
```

保留重复行为，表示用户多次点击 `sku1`，兴趣更强。

适合：

```text
点击序列
浏览序列
搜索序列
```

#### 去重

```text
[sku1, sku2, sku3]
```

避免同一商品反复出现。

适合：

```text
购买序列
收藏序列
长期兴趣序列
```

---

### 5.2.5 行为类型编码

用户行为不仅有商品 ID，还可能有行为类型：

```text
点击、加购、收藏、购买、搜索
```

可以增加行为类型 embedding：

```text
behavior_type_emb = Embedding(behavior_type)
```

例如：

```text
行为序列：
[
  {sku: sku1, action: click},
  {sku: sku2, action: cart},
  {sku: sku3, action: buy}
]
```

每个位置的输入可以是：

```text
position_emb_i = sku_emb_i + action_emb_i
```

或者拼接：

```text
position_emb_i = concat([sku_emb_i, action_emb_i])
```

---

### 5.2.6 时间特征编码

行为发生时间也很重要。

例如：

```text
1 分钟前点击
1 天前点击
30 天前点击
```

兴趣强度可能不同。

常见时间特征：

| 时间特征 | 示例 |
|---|---|
| 距当前时间间隔 | `time_diff = 3600s` |
| 行为发生小时 | `hour = 22` |
| 行为发生星期 | `weekday = 6` |
| 是否节假日 | `is_holiday = 1` |

可以使用：

```text
time_gap_bucket
  ↓
time_gap_embedding
```

例如：

```text
time_gap:
0-5min      -> 1
5-30min     -> 2
30min-2h    -> 3
2h-1d       -> 4
1d-7d       -> 5
7d+         -> 6
```

然后：

```text
position_emb_i = sku_emb_i + action_emb_i + time_gap_emb_i
```

---

## 5.3 行为序列如何输入神经网络？

行为序列首先经过 embedding lookup：

```text
click_sku_seq = [sku1, sku2, sku3, PAD, PAD]
```

查表后：

```text
click_seq_emb.shape = [batch_size, seq_len, embed_dim]
```

例如：

```text
batch_size = 1024
seq_len = 50
embed_dim = 64

click_seq_emb.shape = [1024, 50, 64]
mask.shape = [1024, 50]
```

然后根据模型结构不同，有多种处理方式。

---

# 6. 行为序列的常见建模方式

## 6.1 Mean Pooling / Sum Pooling

最简单方法是对序列 embedding 做平均或求和。

```text
user_interest_emb = mean(click_seq_emb, mask)
```

例如：

```text
click_seq_emb: [batch_size, seq_len, emb_dim]
mask:          [batch_size, seq_len]

输出：
user_interest_emb: [batch_size, emb_dim]
```

优点：

- 简单高效
- 延迟低
- 线上部署方便

缺点：

- 无法区分不同候选商品
- 无法捕捉兴趣多样性
- 对时间顺序不敏感

适合基础模型、低延迟场景。

---

## 6.2 Attention Pooling

Attention 根据候选商品动态计算历史行为的重要性。

例如候选商品是：

```text
candidate_sku = AirPods
```

用户历史：

```text
[奶粉, 手机壳, iPhone, 蓝牙耳机, 充电器]
```

Attention 应该给相关行为更高权重：

```text
蓝牙耳机、iPhone、充电器 权重更高
奶粉 权重更低
```

计算形式：

```text
score_i = attention(candidate_emb, history_emb_i)
weight_i = softmax(score_i)
user_interest_emb = sum(weight_i * history_emb_i)
```

输出：

```text
user_interest_emb.shape = [batch_size, emb_dim]
```

典型模型：

```text
DIN，Deep Interest Network
```

---

## 6.3 DIN：Deep Interest Network

DIN 的核心思想是：

> 用户兴趣不是固定向量，而是应该根据当前候选商品动态激活。

对于候选商品 A：

```text
用户兴趣可能偏向数码
```

对于候选商品 B：

```text
用户兴趣可能偏向母婴
```

DIN 会用候选 item embedding 作为 query，对用户历史 item embedding 做 attention。

输入：

```text
candidate_item_emb: [batch_size, emb_dim]
history_item_emb:   [batch_size, seq_len, emb_dim]
mask:               [batch_size, seq_len]
```

输出：

```text
interest_emb: [batch_size, emb_dim]
```

然后拼接：

```text
model_input = concat([
  user_emb,
  item_emb,
  context_emb,
  dense_features,
  interest_emb
])
```

---

## 6.4 DIEN：Deep Interest Evolution Network

DIEN 在 DIN 基础上进一步建模兴趣演化。

流程：

```text
历史行为 embedding
  ↓
GRU 建模行为序列
  ↓
Attention / AUGRU
  ↓
用户兴趣演化向量
```

适合序列顺序重要的场景，例如：

```text
用户从浏览手机 → 手机壳 → 充电器 → 耳机
```

说明兴趣正在围绕某个主题演化。

---

## 6.5 Transformer / Self-Attention

Transformer 可以捕捉序列内部复杂关系。

输入：

```text
seq_emb = item_emb + position_emb + action_emb + time_emb
```

形状：

```text
[batch_size, seq_len, emb_dim]
```

经过多层 self-attention：

```text
seq_hidden = TransformerEncoder(seq_emb, mask)
```

可以取：

```text
CLS 向量
最后一个位置向量
attention pooling 后的向量
候选 item cross attention 结果
```

优点：

- 表达能力强
- 能建模长序列依赖
- 能捕捉多兴趣模式

缺点：

- 计算开销较大
- 延迟较高
- 对线上精排需要优化

适合高价值精排、重排序、长序列建模。

---

# 7. 多值 Sparse 特征

除了行为序列，还有很多多值离散特征，例如：

```text
item_tags = [自营, 低价, 新品]
user_interest_categories = [手机, 电脑, 家电]
user_preferred_brands = [Apple, Huawei, Xiaomi]
```

处理方式和短序列类似：

```text
multi_value_ids
  ↓
embedding lookup
  ↓
pooling
  ↓
fixed-size vector
```

例如：

```text
item_tags = [tag1, tag2, tag3]
```

查表：

```text
tag_embs.shape = [3, 8]
```

平均池化：

```text
item_tag_emb = mean(tag_embs)
```

得到：

```text
item_tag_emb.shape = [8]
```

---

# 8. 特征融合方式

不同类型特征处理后，需要融合成模型输入。

## 8.1 直接拼接

最常见方式：

```text
model_input = concat([
  dense_features,
  user_emb,
  item_emb,
  category_emb,
  brand_emb,
  shop_emb,
  behavior_interest_emb
])
```

例如：

| 特征向量 | 维度 |
|---|---:|
| dense_features | 20 |
| user_emb | 64 |
| item_emb | 64 |
| brand_emb | 16 |
| category_emb | 16 |
| shop_emb | 32 |
| behavior_interest_emb | 64 |

总维度：

```text
20 + 64 + 64 + 16 + 16 + 32 + 64 = 276
```

输入 DNN：

```text
DNN input shape = [batch_size, 276]
```

---

## 8.2 特征交叉

推荐系统中，特征交叉非常重要。

例如：

```text
user_id × item_id
user_id × category_id
city_id × item_id
gender × category_id
```

深度模型可以自动学习部分交叉，但显式建模通常更有效。

常见方法：

| 方法 | 作用 |
|---|---|
| FM | 二阶特征交叉 |
| DeepFM | FM + DNN |
| DCN | 显式 Cross Network |
| xDeepFM | 高阶显式交叉 |
| AutoInt | 基于 Attention 的特征交互 |
| FiBiNET | 双线性特征交互 |
| PNN | Product-based Neural Network |

---

## 8.3 Field-wise Embedding 输入

很多推荐模型按 field 组织 embedding：

```text
field 1: user_id
field 2: item_id
field 3: brand_id
field 4: category_id
field 5: city_id
```

每个 field 一个 embedding：

```text
sparse_embs.shape = [batch_size, num_fields, embed_dim]
```

例如：

```text
batch_size = 1024
num_fields = 20
embed_dim = 16

sparse_embs.shape = [1024, 20, 16]
```

然后：

- 展平后输入 DNN：

```text
flatten = [1024, 320]
```

- 输入 FM 做二阶交叉：

```text
FM(sparse_embs)
```

- 输入 Attention 做 field 交互：

```text
AutoInt(sparse_embs)
```

---

# 9. 一个完整样本示例

## 9.1 原始样本

假设一个推荐精排样本如下：

```json
{
  "label": 1,
  "user": {
    "user_id": "u123",
    "age": 28,
    "gender": "male",
    "city": "beijing",
    "member_level": "plus",
    "click_7d": 35
  },
  "item": {
    "sku_id": "sku_10099320302155",
    "brand_id": "apple",
    "category_id": "earphone",
    "shop_id": "shop_888",
    "price": 1299.0,
    "ctr_1d": 0.035,
    "sales_30d": 50000
  },
  "context": {
    "hour": 22,
    "device": "ios",
    "scene": "homepage"
  },
  "behavior": {
    "click_sku_seq": ["sku_a", "sku_b", "sku_c"],
    "buy_category_seq": ["phone", "digital"],
    "search_query_seq": ["iphone", "airpods"]
  }
}
```

---

## 9.2 Dense 特征处理

原始 dense：

```text
age = 28
click_7d = 35
price = 1299
ctr_1d = 0.035
sales_30d = 50000
hour = 22
```

处理：

```text
age_norm = (28 - age_mean) / age_std
click_7d_log = log(1 + 35)
price_log = log(1 + 1299)
ctr_1d = 0.035
sales_30d_log = log(1 + 50000)
hour_bucket = 22
```

其中 `hour` 也可以作为类别特征，而不是 dense 特征。

得到：

```text
dense_features = [
  age_norm,
  click_7d_log_norm,
  price_log_norm,
  ctr_1d_norm,
  sales_30d_log_norm
]
```

---

## 9.3 Sparse / Category 特征处理

映射：

```text
user_id      u123                  -> 10001
gender       male                  -> 1
city         beijing               -> 10
member_level plus                  -> 4
sku_id       sku_10099320302155    -> 987654
brand_id     apple                 -> 100
category_id  earphone              -> 200
shop_id      shop_888              -> 888
device       ios                   -> 2
scene        homepage              -> 1
hour         22                    -> 22
```

Embedding：

```text
user_emb       = Embedding(user_id)
gender_emb     = Embedding(gender)
city_emb       = Embedding(city)
member_emb     = Embedding(member_level)
sku_emb        = Embedding(sku_id)
brand_emb      = Embedding(brand_id)
category_emb   = Embedding(category_id)
shop_emb       = Embedding(shop_id)
device_emb     = Embedding(device)
scene_emb      = Embedding(scene)
hour_emb       = Embedding(hour)
```

---

## 9.4 行为序列处理

点击商品序列：

```text
click_sku_seq = ["sku_a", "sku_b", "sku_c"]
```

映射：

```text
click_sku_seq_idx = [101, 102, 103]
```

假设最大长度为 5，padding：

```text
click_sku_seq_idx = [101, 102, 103, 0, 0]
click_mask        = [1,   1,   1,   0, 0]
```

Embedding：

```text
click_seq_emb.shape = [5, 64]
```

使用 attention：

```text
interest_emb = Attention(query=sku_emb, keys=click_seq_emb, mask=click_mask)
```

含义是：

> 用当前候选商品 `sku_10099320302155` 去激活用户历史点击序列中与它最相关的行为。

---

## 9.5 最终模型输入

最终拼接：

```text
model_input = concat([
  dense_features,
  user_emb,
  sku_emb,
  gender_emb,
  city_emb,
  member_emb,
  brand_emb,
  category_emb,
  shop_emb,
  device_emb,
  scene_emb,
  hour_emb,
  interest_emb
])
```

然后送入 DNN：

```text
hidden = DNN(model_input)
ctr_score = sigmoid(Dense(hidden))
```

如果是多任务模型：

```text
ctr_score = sigmoid(Dense_ctr(hidden))
cvr_score = sigmoid(Dense_cvr(hidden))
gmv_score = Dense_gmv(hidden)
```

---

# 10. PyTorch 风格简化示例

下面是一个简化的精排模型结构。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRankModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Sparse / Category embeddings
        self.user_emb = nn.Embedding(10_000_000, 64)
        self.item_emb = nn.Embedding(50_000_000, 64)
        self.brand_emb = nn.Embedding(1_000_000, 16)
        self.category_emb = nn.Embedding(100_000, 16)
        self.city_emb = nn.Embedding(10_000, 8)
        self.device_emb = nn.Embedding(100, 8)

        # DNN
        input_dim = 20 + 64 + 64 + 16 + 16 + 8 + 8 + 64

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def attention_pooling(self, target_emb, seq_emb, mask):
        """
        target_emb: [batch_size, emb_dim]
        seq_emb:    [batch_size, seq_len, emb_dim]
        mask:       [batch_size, seq_len]
        """

        # target 扩展为 [batch_size, seq_len, emb_dim]
        target = target_emb.unsqueeze(1).expand_as(seq_emb)

        # 简单 dot attention
        scores = torch.sum(target * seq_emb, dim=-1)  # [batch_size, seq_len]

        # mask padding 位置
        scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len]

        # 加权求和
        interest_emb = torch.sum(seq_emb * weights.unsqueeze(-1), dim=1)

        return interest_emb

    def forward(self, features):
        dense = features["dense"]  # [batch_size, 20]

        user_id = features["user_id"]
        item_id = features["item_id"]
        brand_id = features["brand_id"]
        category_id = features["category_id"]
        city_id = features["city_id"]
        device_id = features["device_id"]

        click_seq = features["click_seq"]      # [batch_size, seq_len]
        click_mask = features["click_mask"]    # [batch_size, seq_len]

        user_emb = self.user_emb(user_id)
        item_emb = self.item_emb(item_id)
        brand_emb = self.brand_emb(brand_id)
        category_emb = self.category_emb(category_id)
        city_emb = self.city_emb(city_id)
        device_emb = self.device_emb(device_id)

        click_seq_emb = self.item_emb(click_seq)

        interest_emb = self.attention_pooling(
            target_emb=item_emb,
            seq_emb=click_seq_emb,
            mask=click_mask
        )

        x = torch.cat([
            dense,
            user_emb,
            item_emb,
            brand_emb,
            category_emb,
            city_emb,
            device_emb,
            interest_emb
        ], dim=-1)

        logit = self.mlp(x)
        ctr = torch.sigmoid(logit)

        return ctr
```

---

# 11. Tensor 形状示例

假设：

```text
batch_size = 1024
dense_dim = 20
embed_dim = 64
seq_len = 50
```

输入张量可能是：

| 特征 | Shape | 说明 |
|---|---|---|
| `dense` | `[1024, 20]` | 连续特征 |
| `user_id` | `[1024]` | 用户 ID index |
| `item_id` | `[1024]` | 商品 ID index |
| `brand_id` | `[1024]` | 品牌 ID index |
| `category_id` | `[1024]` | 类目 ID index |
| `click_seq` | `[1024, 50]` | 点击商品序列 |
| `click_mask` | `[1024, 50]` | 序列 mask |

Embedding 后：

| 特征 | Shape |
|---|---|
| `user_emb` | `[1024, 64]` |
| `item_emb` | `[1024, 64]` |
| `brand_emb` | `[1024, 16]` |
| `category_emb` | `[1024, 16]` |
| `click_seq_emb` | `[1024, 50, 64]` |
| `interest_emb` | `[1024, 64]` |

最终拼接：

```text
x.shape = [1024, total_input_dim]
```

---

# 12. 训练与线上推理的一致性

在推荐系统中，特征预处理不仅要关注训练，还要关注线上服务。

最重要原则是：

> **训练和线上必须使用完全一致的特征处理逻辑。**

否则会出现 training-serving skew。

例如：

| 问题 | 后果 |
|---|---|
| 训练时价格做了 log，线上没做 | 分布不一致，模型效果下降 |
| 训练词表和线上词表不一致 | Embedding 查错 |
| 训练样本有未来信息 | 线上无法复现，离线效果虚高 |
| 训练时序列按近到远，线上按远到近 | 序列模型语义错乱 |
| 训练 padding 在右侧，线上 padding 在左侧 | mask 或位置编码错误 |

---

# 13. 推荐系统中特征预处理常见工程实践

## 13.1 离线特征

用于训练样本构建，通常来自：

```text
Hive / Spark / Flink / Kafka / HDFS
```

特点：

- 数据量大
- 可计算长周期统计特征
- 可生成训练样本
- 可做复杂 join

例如：

```text
用户近 7 天点击次数
商品近 30 天销量
用户-类目近 14 天点击次数
```

---

## 13.2 在线特征

用于实时推理，通常来自：

```text
Redis / KV / Feature Store / 实时流计算系统
```

特点：

- 延迟敏感
- 需要高可用
- 特征不能太复杂
- 要与训练逻辑对齐

例如：

```text
用户最近 50 个点击商品
用户实时会话行为
商品实时库存
实时价格
```

---

## 13.3 特征注册与元信息

大型精排系统通常会维护特征元信息：

```json
{
  "feature_name": "item_price",
  "feature_type": "dense",
  "default_value": 0,
  "normalization": "log_standard",
  "mean": 4.2,
  "std": 1.1
}
```

对于 sparse 特征：

```json
{
  "feature_name": "sku_id",
  "feature_type": "sparse",
  "vocab_type": "hash",
  "bucket_size": 50000000,
  "embedding_dim": 64,
  "oov_id": 0
}
```

---

# 14. 各类特征处理方式总结

| 特征类型 | 原始形式 | 预处理 | 输入模型方式 |
|---|---|---|---|
| Dense 特征 | 连续数值 | 缺失填充、截断、log、标准化 | 直接拼接成 dense tensor |
| Sparse ID 特征 | 用户 ID、商品 ID | vocab/hash、OOV 处理 | Embedding lookup |
| Category 特征 | 性别、城市、设备 | label encoding / one-hot / embedding | one-hot 或 embedding |
| Multi-value 特征 | 标签、兴趣类目 | ID 映射、padding、pooling | Embedding + pooling |
| 行为序列特征 | 点击/购买序列 | 排序、截断、padding、mask | Embedding + pooling/attention/Transformer |
| 统计交叉特征 | 用户-类目点击数 | 平滑、归一化、时间窗口 | dense 输入或 embedding |
| 上下文特征 | 时间、场景、设备 | 分桶、类别编码 | embedding 或 dense |

---

# 15. 核心要点

1. **Dense 特征**要重点处理数值尺度问题，常用缺失值填充、截断、log、标准化。
2. **Sparse 特征**不能直接把 ID 当数值输入，通常需要映射为 index 后做 **Embedding lookup**。
3. **Category 特征**本质上是离散特征，小类别可 one-hot，大类别更适合 embedding。
4. **用户行为序列特征**需要排序、截断、padding、mask，再通过 pooling、attention、DIN、DIEN 或 Transformer 建模。
5. **最终输入 DNN 的不是原始业务字段，而是 dense 向量、embedding 向量和序列编码向量的融合结果**。
6. **训练和线上特征处理必须一致**，否则模型离线效果和线上效果会严重偏离。
7. 在精排模型中，最常见的最终结构是：

```text
dense features
+ user/item/context embeddings
+ behavior sequence interest embeddings
+ feature interaction modules
→ DNN
→ CTR/CVR/GMV score
```

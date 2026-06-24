# 推荐模型中 Sparse 特征哈希分桶与 Embedding 映射说明

推荐算法模型里，**user_id、item_id、性别、年龄分段** 都可以被视为 sparse/categorical 特征，但它们的处理方式不完全一样。

核心原则是：

> **高基数 sparse 特征**，如 user_id、item_id，通常需要哈希分桶或大规模 ID 映射后再查 embedding。  
> **低基数 sparse 特征**，如性别、年龄分段，通常不需要哈希分桶，直接建立小词表映射到 embedding 即可。

---

# 1. 整体处理流程

推荐模型中的 sparse 特征一般经过如下链路：

```text
原始特征
  ↓
特征清洗 / 规范化
  ↓
类别编码 / 哈希分桶
  ↓
得到离散 ID / bucket_id
  ↓
查 Embedding Table
  ↓
得到 dense embedding vector
  ↓
送入 DNN / FM / DeepFM / DIN / DSSM / Transformer 等模型
```

以一个样本为例：

```text
user_id = "u_123456789"
item_id = "sku_987654321"
gender = "男"
age = 27
```

处理后可能变成：

```text
user_id_bucket = 3489123
item_id_bucket = 87654321
gender_id = 1
age_bucket = 2
```

然后分别查 embedding：

```text
user_emb = UserEmbeddingTable[3489123]
item_emb = ItemEmbeddingTable[87654321]
gender_emb = GenderEmbeddingTable[1]
age_emb = AgeEmbeddingTable[2]
```

---

# 2. 什么是哈希分桶

## 2.1 哈希分桶的定义

**哈希分桶**是指用哈希函数把原始类别特征映射到一个固定范围内的整数桶：

```text
bucket_id = Hash(feature_value) % bucket_size
```

例如：

```text
bucket_id = MurmurHash3("user_id=u_123456789") % 10,000,000
```

如果结果是：

```text
3489123
```

那么这个 user_id 就会映射到第 `3489123` 个 embedding 向量。

---

## 2.2 哈希分桶的目的

哈希分桶主要解决 **高基数类别特征** 的几个问题：

| 问题 | 说明 |
|---|---|
| **ID 数量巨大** | user_id、item_id 可能是千万、亿级甚至十亿级，无法无限扩张 embedding table |
| **新 ID 不断出现** | 新用户、新商品每天产生，如果用静态词表，更新成本高 |
| **内存不可控** | 每个 ID 都分配 embedding，会导致参数量巨大 |
| **训练/线上一致性** | 哈希函数可以在线上实时计算，避免依赖庞大词表服务 |
| **冷启动兼容** | 新 ID 即使没见过，也能被哈希到某个桶中，虽然可能与其他 ID 冲突 |

---

# 3. 常用哈希算法

推荐系统中常见的哈希算法包括：

| 哈希算法 | 特点 | 是否常用 |
|---|---|---|
| **MurmurHash / MurmurHash3** | 速度快，分布较均匀，工程中非常常见 | **非常常用** |
| **CityHash** | Google 提出的高性能字符串哈希 | 常用 |
| **FarmHash** | CityHash 的后续版本，性能较好 | 常用 |
| **xxHash** | 极快，适合高吞吐场景 | 常用 |
| **FNV Hash** | 实现简单，速度快 | 一般 |
| **CRC32** | 硬件友好，但分布性不一定最优 | 一般 |
| **MD5 / SHA 系列** | 分布均匀但计算较重，更多用于安全/去重，不是推荐主流选择 | 不建议作为高频特征哈希 |

推荐系统线上高并发场景，一般优先考虑：

```text
MurmurHash3、CityHash、FarmHash、xxHash
```

不建议使用语言内置的非稳定哈希，例如 Python 的 `hash()`，因为它在不同进程、不同启动周期中可能不稳定，不适合训练和线上一致性要求高的场景。

---

# 4. user_id 如何处理

## 4.1 user_id 的特点

`user_id` 是典型的 **高基数 sparse 特征**：

- 用户数量可能是千万、亿级甚至十亿级；
- 用户活跃度长尾严重；
- 新用户不断产生；
- 直接 one-hot 维度过大；
- 直接为每个 user_id 建 embedding table，参数量和存储成本很高。

---

## 4.2 常见处理方式

### 方式一：哈希分桶

```text
user_bucket_id = Hash("user_id=" + user_id) % user_bucket_size
```

例如：

```text
user_id = "u_123456789"
user_bucket_size = 10,000,000

user_bucket_id = MurmurHash3("user_id=u_123456789") % 10,000,000
               = 3,489,123
```

然后查表：

```text
user_emb = UserEmbeddingTable[3,489,123]
```

---

## 4.3 user_id 的 embedding table

假设：

```text
user_bucket_size = 10,000,000
embedding_dim = 32
```

则：

```text
UserEmbeddingTable.shape = [10,000,000, 32]
```

### key 和 value 分别是什么？

| 对象 | 含义 |
|---|---|
| **key** | 哈希后的 `user_bucket_id`，例如 `3,489,123` |
| **value** | 对应的 embedding 向量，例如 32 维浮点向量 |

例如：

```text
key = 3,489,123

value = [
  0.012, -0.087, 0.331, ..., 0.045
]
```

---

## 4.4 user_id 哈希分桶的结果是什么？

哈希分桶的结果是一个整数桶 ID：

```text
user_id = "u_123456789"
↓
MurmurHash3
↓
hash_value = 128739847129837
↓
hash_value % 10,000,000
↓
user_bucket_id = 3,489,123
```

最终模型并不知道原始 user_id，而是使用：

```text
UserEmbeddingTable[3,489,123]
```

---

## 4.5 user_id 处理的注意点

### 1. 哈希冲突

不同 user_id 可能映射到同一个 bucket：

```text
Hash("u_123") % N = 10086
Hash("u_456") % N = 10086
```

这两个用户会共享同一个 embedding。

这是哈希分桶的主要代价。

---

### 2. bucket_size 需要足够大

如果用户量很大，bucket_size 太小会导致严重冲突。

例如：

| 用户规模 | bucket_size 建议 |
|---|---|
| 百万级 | 百万到千万级 |
| 千万级 | 千万到亿级 |
| 亿级 | 亿级或采用分层/参数服务器方案 |

实际大小需要结合：

- 可用内存；
- embedding 维度；
- 用户活跃度；
- 冲突容忍度；
- 模型收益；
- 线上延迟。

---

### 3. 可结合频次过滤

对极低频用户，不一定需要单独建强表达 embedding，可以：

- 映射到 hash bucket；
- 映射到 OOV bucket；
- 使用用户画像特征替代；
- 使用行为序列建模；
- 对高频用户建 ID embedding，低频用户共享默认 embedding。

---

# 5. item_id 如何处理

## 5.1 item_id 的特点

`item_id` 也是典型高基数 sparse 特征，尤其在电商、内容、广告场景中非常重要。

例如：

- 电商：sku_id、spu_id、shop_id、brand_id；
- 内容：video_id、article_id、author_id；
- 广告：ad_id、creative_id、campaign_id。

item_id 的特点是：

| 特点 | 说明 |
|---|---|
| **数量大** | 商品/内容/广告素材可能达到千万、亿级 |
| **更新快** | 新商品、新内容、新素材不断上线 |
| **长尾明显** | 大量 item 曝光和点击很少 |
| **对推荐强相关** | item embedding 往往是推荐模型的核心参数 |

---

## 5.2 item_id 的处理方式

### 方式一：哈希分桶

```text
item_bucket_id = Hash("item_id=" + item_id) % item_bucket_size
```

例如：

```text
item_id = "sku_987654321"
item_bucket_size = 50,000,000

item_bucket_id = MurmurHash3("item_id=sku_987654321") % 50,000,000
               = 18,765,432
```

查表：

```text
item_emb = ItemEmbeddingTable[18,765,432]
```

---

### 方式二：词表 ID 映射

如果系统可以维护大规模 item 词表，也可以做：

```text
item_id -> item_index
```

例如：

```text
sku_987654321 -> 18,765,432
```

然后：

```text
item_emb = ItemEmbeddingTable[18,765,432]
```

这种方式没有哈希冲突，但需要维护词表，并处理新增 item。

---

### 方式三：高频 item 独立建表，低频 item 哈希/OOV

工业系统里经常采用混合方案：

```text
高频 item：独立 ID 映射
低频 item：哈希分桶
未知 item：OOV bucket
```

这样可以兼顾：

- 高频 item 表达能力；
- 低频 item 存储成本；
- 新 item 在线兼容性。

---

## 5.3 item_id 的 embedding table

假设：

```text
item_bucket_size = 50,000,000
embedding_dim = 64
```

则：

```text
ItemEmbeddingTable.shape = [50,000,000, 64]
```

### key 和 value 分别是什么？

| 对象 | 含义 |
|---|---|
| **key** | 哈希后的 `item_bucket_id` 或词表映射后的 `item_index` |
| **value** | 该 item 对应的 dense embedding 向量 |

例如：

```text
key = 18,765,432

value = [
  0.102, -0.044, 0.251, ..., 0.019
]
```

---

# 6. 性别 gender 如何处理

## 6.1 gender 的特点

性别是 **低基数 sparse 特征**，类别数很少。

例如：

```text
男、女、未知
```

也可能包括：

```text
male、female、unknown
```

这类特征没有必要使用大规模哈希分桶。

---

## 6.2 推荐处理方式：直接词表映射

可以建立一个小词表：

```text
{
  "未知": 0,
  "男": 1,
  "女": 2
}
```

样本：

```text
gender = "男"
```

映射为：

```text
gender_id = 1
```

然后查表：

```text
gender_emb = GenderEmbeddingTable[1]
```

---

## 6.3 gender 的 embedding table

假设：

```text
gender_vocab_size = 3
embedding_dim = 4
```

则：

```text
GenderEmbeddingTable.shape = [3, 4]
```

### key 和 value 分别是什么？

| 对象 | 含义 |
|---|---|
| **key** | 性别类别 ID，例如 `0/1/2` |
| **value** | 对应的 embedding 向量 |

例如：

```text
key = 1

value = [0.21, -0.08, 0.13, 0.04]
```

---

## 6.4 性别需要哈希分桶吗？

**通常不需要。**

原因：

1. **类别数太少**，直接映射即可；
2. 哈希分桶没有节省明显成本；
3. 哈希可能引入不必要的冲突；
4. 直接词表更可解释、更稳定；
5. 线上和离线维护成本低。

当然，某些统一特征处理框架可能会把所有 sparse 特征都走统一 hash pipeline，例如：

```text
gender_bucket_id = Hash("gender=男") % 100
```

这在工程上可行，但从建模角度看不是必要的。

---

# 7. 年龄分段 age_bucket 如何处理

## 7.1 年龄可以先离散化

年龄本身是数值特征，可以有两种处理方式：

### 方式一：作为连续数值特征

例如：

```text
age = 27
```

做归一化：

```text
age_norm = age / 100
```

作为 dense feature 输入模型。

---

### 方式二：分桶后作为 sparse 特征

用户问题中提到按 10 岁一段，例如：

| 年龄范围 | age_bucket |
|---|---|
| 未知 | 0 |
| 0-9 | 1 |
| 10-19 | 2 |
| 20-29 | 3 |
| 30-39 | 4 |
| 40-49 | 5 |
| 50-59 | 6 |
| 60-69 | 7 |
| 70-79 | 8 |
| 80+ | 9 |

如果：

```text
age = 27
```

则：

```text
age_bucket = 3
```

然后查表：

```text
age_emb = AgeEmbeddingTable[3]
```

---

## 7.2 age_bucket 的 embedding table

假设：

```text
age_bucket_num = 10
embedding_dim = 4
```

则：

```text
AgeEmbeddingTable.shape = [10, 4]
```

### key 和 value 分别是什么？

| 对象 | 含义 |
|---|---|
| **key** | 年龄分段 ID，例如 `3` 表示 20-29 岁 |
| **value** | 对应年龄段的 embedding 向量 |

例如：

```text
key = 3

value = [0.05, 0.18, -0.11, 0.07]
```

---

## 7.3 年龄分段需要哈希分桶吗？

**通常不需要。**

原因和性别类似：

1. 年龄分段类别数很少；
2. 直接映射最简单；
3. 不存在高基数 ID 膨胀问题；
4. 不需要通过哈希控制 embedding table 大小；
5. 哈希会降低可解释性。

推荐方式是：

```text
age → age_bucket → age_bucket_id → age_embedding
```

而不是：

```text
age_bucket → hash(age_bucket) % bucket_size
```

---

# 8. 四类特征处理方式对比

| 特征 | 类别数 | 是否高基数 | 推荐处理方式 | 是否需要哈希分桶 | Embedding Table Key | Embedding Table Value |
|---|---:|---|---|---|---|---|
| **user_id** | 千万/亿级 | 是 | 哈希分桶或大规模 ID 映射 | **通常需要** | `user_bucket_id` / `user_index` | 用户 embedding 向量 |
| **item_id** | 千万/亿级 | 是 | 哈希分桶、词表映射或混合方案 | **通常需要** | `item_bucket_id` / `item_index` | 商品 embedding 向量 |
| **gender** | 2-3 类 | 否 | 小词表直接映射 | **通常不需要** | `gender_id` | 性别 embedding 向量 |
| **age_bucket** | 约 10 类 | 否 | 分桶后直接映射 | **通常不需要** | `age_bucket_id` | 年龄段 embedding 向量 |

---

# 9. Embedding Table 的 key 和 value 到底是什么

## 9.1 通用形式

Embedding table 可以理解为一个矩阵：

```text
EmbeddingTable = [
  emb_0,
  emb_1,
  emb_2,
  ...
  emb_N
]
```

其中：

```text
key = row_id
value = embedding vector
```

例如：

```text
key = 10086
value = [0.12, -0.03, 0.88, ..., 0.09]
```

---

## 9.2 对于哈希分桶特征

以 user_id 为例：

```text
原始特征值: user_id = "u_123456789"
哈希结果: bucket_id = 3,489,123
Embedding Table key: 3,489,123
Embedding Table value: 第 3,489,123 行 embedding 向量
```

即：

```text
user_emb = UserEmbeddingTable[3,489,123]
```

---

## 9.3 对于低基数特征

以 gender 为例：

```text
原始特征值: gender = "男"
词表映射: gender_id = 1
Embedding Table key: 1
Embedding Table value: 第 1 行 embedding 向量
```

即：

```text
gender_emb = GenderEmbeddingTable[1]
```

---

# 10. 是否所有 sparse 特征都要哈希分桶？

**不是。**

需要区分：

## 10.1 适合哈希分桶的特征

适合哈希分桶的特征通常具有以下特点：

- 类别数量巨大；
- 新类别持续出现；
- 词表维护成本高；
- 长尾严重；
- 可接受一定哈希冲突；
- embedding table 参数量需要受控。

典型特征：

```text
user_id
item_id
shop_id
seller_id
brand_id
query
tag
author_id
creative_id
campaign_id
device_id
```

其中，并不是所有都必须 hash，而是要根据规模和稳定性选择。

---

## 10.2 不适合或不需要哈希分桶的特征

低基数特征通常不需要哈希分桶：

```text
gender
age_bucket
city_level
membership_level
device_type
os_type
network_type
weekday
hour
```

这些特征更适合：

```text
固定词表映射 + embedding
```

或者直接作为 one-hot / dense feature 输入。

---

# 11. 哈希分桶与词表映射的区别

| 维度 | 哈希分桶 | 词表映射 |
|---|---|---|
| 映射方式 | `Hash(value) % N` | 查词表：`value -> index` |
| 是否需要维护词表 | 不一定需要 | 需要 |
| 是否有冲突 | **有可能冲突** | 通常无冲突 |
| 新 ID 处理 | 天然支持 | 需要 OOV 或更新词表 |
| 可解释性 | 较弱 | 较强 |
| 适用特征 | 高基数、动态 ID | 低基数、稳定类别、高频 ID |
| 工程复杂度 | 在线计算简单 | 依赖词表服务/版本管理 |
| 参数规模控制 | 通过 bucket_size 控制 | 由词表规模控制 |

---

# 12. 是否可以把 feature_name 一起参与 hash？

**建议参与。**

原因是避免不同特征域之间发生不必要的冲突。

例如：

```text
user_id = "123"
item_id = "123"
```

如果只 hash `"123"`，可能得到相同 bucket。

更合理的做法是：

```text
Hash("user_id=123") % user_bucket_size
Hash("item_id=123") % item_bucket_size
```

或者：

```text
Hash(feature_name + ":" + feature_value) % bucket_size
```

这样可以区分不同特征域。

---

# 13. 不同特征是否共用 embedding table？

有两种设计。

## 13.1 每个特征域单独一张表

例如：

```text
UserEmbeddingTable
ItemEmbeddingTable
GenderEmbeddingTable
AgeEmbeddingTable
```

优点：

- 不同特征可以使用不同 embedding 维度；
- 不同特征不会互相冲突；
- 便于单独调参；
- 便于控制参数规模。

这是推荐系统中很常见的方式。

---

## 13.2 多个特征域共用一张大表

例如：

```text
GlobalEmbeddingTable
```

所有 sparse 特征都映射到同一个全局 ID 空间：

```text
global_id = Hash(feature_name + "=" + feature_value) % global_bucket_size
```

优点：

- 工程统一；
- 方便通用化特征处理；
- 对超大规模稀疏特征比较友好。

缺点：

- 不同特征之间可能冲突；
- 低基数特征也可能受冲突影响；
- 不同特征无法灵活设置 embedding 维度；
- 可解释性较弱。

---

# 14. 实际例子：四个特征完整处理

假设样本：

```text
user_id = "u_123456789"
item_id = "sku_987654321"
gender = "男"
age = 27
```

## 14.1 user_id

```text
user_bucket_size = 10,000,000
user_embedding_dim = 32

user_bucket_id = MurmurHash3("user_id=u_123456789") % 10,000,000
               = 3,489,123

user_emb = UserEmbeddingTable[3,489,123]
```

Embedding table：

```text
UserEmbeddingTable.shape = [10,000,000, 32]
```

---

## 14.2 item_id

```text
item_bucket_size = 50,000,000
item_embedding_dim = 64

item_bucket_id = MurmurHash3("item_id=sku_987654321") % 50,000,000
               = 18,765,432

item_emb = ItemEmbeddingTable[18,765,432]
```

Embedding table：

```text
ItemEmbeddingTable.shape = [50,000,000, 64]
```

---

## 14.3 gender

```text
gender_vocab = {
  "未知": 0,
  "男": 1,
  "女": 2
}

gender = "男"
gender_id = 1

gender_emb = GenderEmbeddingTable[1]
```

Embedding table：

```text
GenderEmbeddingTable.shape = [3, 4]
```

---

## 14.4 age_bucket

```text
age = 27
age_bucket = "20-29"
age_bucket_id = 3

age_emb = AgeEmbeddingTable[3]
```

Embedding table：

```text
AgeEmbeddingTable.shape = [10, 4]
```

---

# 15. 工程实践建议

## 15.1 对 user_id

建议：

- 如果用户规模很大，优先使用 **哈希分桶** 或 **参数服务器 + 动态 embedding**；
- 高频用户可以独立保留；
- 低频用户可以 hash 或共享 OOV；
- 注意哈希冲突和内存成本；
- 线上和离线必须使用完全一致的 hash 函数和 bucket_size。

---

## 15.2 对 item_id

建议：

- 高频 item 建独立 embedding；
- 低频 item 可 hash；
- 新 item 需要冷启动特征，如类目、品牌、价格、文本、图片 embedding；
- item embedding 通常比 gender、age 等特征更重要，维度可以更高；
- 注意 item 更新频率和下架问题。

---

## 15.3 对 gender、age_bucket

建议：

- 直接用固定词表；
- embedding 维度不宜过大；
- 加入 unknown 类别；
- 不需要大规模 hash bucket；
- 保持可解释性和稳定性。

---

# 16. 总结

## 16.1 核心答案

1. **user_id、item_id 量级很大**，通常需要通过 **哈希分桶** 或 **大规模 ID 映射** 得到整数 ID，再查 embedding。
2. 哈希公式通常是：

```text
bucket_id = Hash(feature_name + "=" + feature_value) % bucket_size
```

3. 常用哈希算法包括：

```text
MurmurHash3、CityHash、FarmHash、xxHash
```

4. 哈希分桶结果是一个整数：

```text
bucket_id ∈ [0, bucket_size - 1]
```

5. Embedding table 中：

```text
key = bucket_id / vocab_id
value = dense embedding vector
```

6. **性别、年龄分段这种低基数 sparse 特征，通常不需要哈希分桶**，直接做固定词表映射即可。
7. 低基数特征强行哈希虽然工程上可行，但建模收益有限，还可能引入不必要的冲突和可解释性损失。

## 16.2 推荐配置

| 特征 | 推荐方式 |
|---|---|
| **user_id** | `Hash(user_id) % user_bucket_size` 或高频 ID 映射 + 低频 hash |
| **item_id** | `Hash(item_id) % item_bucket_size` 或 item 词表 + OOV/hash |
| **gender** | 固定词表：未知/男/女 |
| **age_bucket** | 年龄离散化后固定词表映射 |

最终可表示为：

```text
[user_emb, item_emb, gender_emb, age_emb]
        ↓
concat / pooling / attention / feature interaction
        ↓
推荐模型主体
        ↓
CTR / CVR / GMV / 排序分数
```

KMeans 的处理逻辑可以概括为一句话：

> **先随机选 K 个中心点，然后反复执行“样本分配到最近中心”与“重新计算中心”两个步骤，直到中心基本不再变化。**

它是一种典型的 **无监督聚类算法**。

---

# 1. KMeans 要解决什么问题？

给定一批向量，例如：

```text
10w 个 embedding
```

每个 embedding 是一个高维向量：

```text
x1, x2, x3, ..., xn
```

KMeans 希望把它们分成 `K` 个簇：

```text
cluster 0
cluster 1
cluster 2
...
cluster K-1
```

使得：

> **同一个簇内的向量尽量相似，不同簇之间尽量不同。**

---

# 2. KMeans 的核心目标

KMeans 优化的目标是：让每个样本到自己所属簇中心的距离之和尽量小。

数学上通常写成：

```text
min Σ ||xi - c_label_i||²
```

其中：

| 符号 | 含义 |
|---|---|
| `xi` | 第 i 个 embedding |
| `label_i` | 第 i 个 embedding 所属的簇 id |
| `c_label_i` | 这个簇的中心向量 |
| `K` | 聚类数量 |
| `c0, c1, ..., cK-1` | K 个簇中心 |

直观理解：

> 每个点都找离自己最近的中心，中心再根据分给自己的点重新调整位置。

---

# 3. KMeans 的完整处理流程

假设你有：

```text
n = 100000 个 embedding
K = 100 个簇
```

KMeans 的流程如下。

---

## Step 1：选择 K 个初始中心

先初始化 `K` 个中心点：

```text
center_0
center_1
...
center_99
```

初始化方式常见有两种：

### 1. 随机初始化

从所有 embedding 中随机选 K 个作为初始中心。

缺点是：

```text
不同随机种子可能得到不同结果
```

### 2. KMeans++ 初始化

更常用。

它会让初始中心尽量分散，通常比纯随机更稳定。

在 sklearn 中默认一般就是类似 KMeans++ 的初始化方式：

```python
init="k-means++"
```

---

## Step 2：给每个 embedding 分配最近的中心

对每一个 embedding，计算它和所有中心的距离。

例如第 `i` 个 embedding：

```text
embedding_i
```

分别计算：

```text
dist(embedding_i, center_0)
dist(embedding_i, center_1)
dist(embedding_i, center_2)
...
dist(embedding_i, center_99)
```

然后选择距离最近的中心。

例如：

```text
embedding_i 距离 center_7 最近
```

那么：

```python
label[i] = 7
```

这表示：

> 第 i 个 embedding 属于第 7 个簇。

所以在 KMeans 结果中：

```python
labels[i]
```

就是第 `i` 个 embedding 对应的类簇中心 id。

---

## Step 3：根据分配结果重新计算每个簇中心

假设当前所有属于 `cluster 7` 的 embedding 有：

```text
x10, x25, x108, x999, ...
```

那么新的 `center_7` 就是这些向量的均值：

```text
center_7 = mean(x10, x25, x108, x999, ...)
```

也就是：

```text
把同一个簇里的所有点求平均，得到新的中心点。
```

---

## Step 4：重复 Step 2 和 Step 3

不断重复：

```text
1. 每个样本分配到最近中心
2. 重新计算每个簇中心
```

直到满足停止条件。

常见停止条件包括：

```text
1. 簇中心变化很小
2. 样本分配结果基本不变
3. 达到最大迭代次数 max_iter
```

---

# 4. 图示理解

假设有一些二维点：

```text
  x   x        o o
 x x x        o o o

      *        *
```

其中 `*` 是两个初始中心。

第一次分配：

```text
左边的点分给左中心
右边的点分给右中心
```

然后重新计算中心：

```text
左中心移动到左边点群的平均位置
右中心移动到右边点群的平均位置
```

反复几轮后，中心稳定下来。

---

# 5. KMeans 的伪代码

```text
输入：
    X: n 个 embedding
    K: 聚类数量

初始化：
    随机选择 K 个中心 centers

重复直到收敛：
    1. 对每个样本 xi：
        找到距离 xi 最近的中心 cj
        label[i] = j

    2. 对每个簇 j：
        找出所有 label[i] == j 的样本
        centers[j] = 这些样本的均值

输出：
    labels: 每个样本所属簇 id
    centers: 每个簇的中心向量
```

---

# 6. Python 代码对应处理逻辑

以 sklearn 为例：

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=100,
    init="k-means++",
    max_iter=300,
    random_state=42
)

labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
```

这几行代码背后做的事情就是：

```text
1. 初始化 100 个中心
2. 计算每个 embedding 离哪个中心最近
3. 分配 cluster_id
4. 根据分配结果更新中心
5. 重复直到收敛
```

其中：

```python
labels[i]
```

表示：

```text
第 i 个 embedding 属于哪个簇
```

例如：

```python
labels[0] = 12
```

说明：

```text
第 0 个 embedding 属于 cluster 12
```

对应中心向量是：

```python
centers[12]
```

---

# 7. 用 embedding 聚类时的处理逻辑

对于 embedding，KMeans 通常这么处理：

```text
原始文本/商品/用户
        ↓
embedding 模型编码
        ↓
得到向量矩阵 X
        ↓
可选：L2 normalize
        ↓
KMeans 聚类
        ↓
得到 labels 和 centers
```

其中：

```python
X.shape = [100000, dim]
```

例如：

```python
X.shape = [100000, 1024]
```

表示有 10w 条 embedding，每条 embedding 是 1024 维。

---

# 8. 是否需要 normalize？

如果 embedding 用的是语义相似度，通常建议做 L2 归一化：

```python
from sklearn.preprocessing import normalize

X = normalize(X, norm="l2")
```

原因是：

> 文本 embedding 通常更关心 cosine similarity，而不是原始欧氏距离。

KMeans 默认基于欧氏距离。

当向量被 L2 normalize 后，欧氏距离和 cosine similarity 有较强对应关系：

```text
||a - b||² = 2 - 2cos(a, b)
```

所以：

```text
归一化后的 KMeans 更接近基于 cosine 的聚类效果。
```

---

# 9. KMeans 输出结果怎么看？

KMeans 通常输出两个核心结果：

## 9.1 `labels`

```python
labels = kmeans.labels_
```

或：

```python
labels = kmeans.fit_predict(X)
```

含义：

```text
labels[i] = 第 i 个 embedding 所属的簇 id
```

例如：

```text
labels = [2, 0, 2, 1, 1, 0]
```

表示：

| embedding 下标 | cluster_id |
|---:|---:|
| 0 | 2 |
| 1 | 0 |
| 2 | 2 |
| 3 | 1 |
| 4 | 1 |
| 5 | 0 |

---

## 9.2 `cluster_centers_`

```python
centers = kmeans.cluster_centers_
```

含义：

```text
centers[j] = 第 j 个簇的中心向量
```

例如：

```python
centers[2]
```

表示：

```text
cluster 2 的中心向量
```

所以第 `i` 个 embedding 对应的中心向量是：

```python
center_vector = centers[labels[i]]
```

---

# 10. 每个 embedding 如何找到对应的类簇中心 id？

非常直接：

```python
cluster_id = labels[i]
```

完整例子：

```python
i = 0

cluster_id = labels[i]
center_vector = centers[cluster_id]

print("第 i 个 embedding 的类簇中心 id:", cluster_id)
print("对应中心向量:", center_vector)
```

所以：

```text
每个 embedding 对应的类簇中心 id = labels[i]
```

---

# 11. KMeans 中心是不是原始 embedding？

不一定。

这是一个很重要的点。

KMeans 的中心是：

```text
簇内所有 embedding 的均值向量
```

所以它通常不是某一个真实存在的原始 embedding。

例如某个簇有三个向量：

```text
[1, 1]
[2, 2]
[3, 3]
```

中心是：

```text
[2, 2]
```

这个中心刚好可能存在。

但如果是：

```text
[1, 1]
[2, 3]
[4, 5]
```

中心是：

```text
[2.33, 3.0]
```

它就不是原始样本。

---

# 12. 如果想找每个簇最具代表性的原始 embedding

可以找：

```text
距离簇中心最近的原始 embedding
```

代码：

```python
from sklearn.metrics import pairwise_distances_argmin_min

nearest_indices, distances = pairwise_distances_argmin_min(
    centers,
    X,
    metric="euclidean"
)

for cluster_id, idx in enumerate(nearest_indices):
    print(
        "cluster_id:", cluster_id,
        "representative_embedding_index:", idx,
        "distance:", distances[cluster_id]
    )
```

含义：

```python
nearest_indices[j]
```

表示：

```text
距离第 j 个簇中心最近的原始 embedding 下标
```

这个样本可以作为该簇的代表样本。

---

# 13. MiniBatchKMeans 的处理逻辑

如果你处理 10w embedding，我之前推荐了：

```python
MiniBatchKMeans
```

它和标准 KMeans 的区别是：

## 标准 KMeans

每一轮都用全部数据更新中心：

```text
每轮使用全部 10w 条 embedding
```

优点：

```text
结果更稳定
```

缺点：

```text
数据量大时比较慢
```

---

## MiniBatchKMeans

每次只抽一小批数据更新中心：

```text
每次用 batch_size 条数据
```

例如：

```python
batch_size=4096
```

处理逻辑是：

```text
1. 初始化 K 个中心
2. 随机取一批 embedding
3. 分配到最近中心
4. 用这一批数据更新中心
5. 再取下一批
6. 反复迭代
```

优点：

```text
更快
更省内存
适合 10w、百万级数据
```

缺点：

```text
结果可能略有随机波动
```

---

# 14. KMeans 的复杂度

标准 KMeans 每轮复杂度大约是：

```text
O(n * K * d)
```

其中：

| 符号 | 含义 |
|---|---|
| `n` | 样本数量 |
| `K` | 簇数量 |
| `d` | embedding 维度 |

如果迭代 `T` 轮，总复杂度：

```text
O(T * n * K * d)
```

例如：

```text
n = 100000
K = 100
d = 1024
T = 100
```

计算量就比较大，所以高维大规模 embedding 时，FAISS 或 MiniBatchKMeans 更合适。

---

# 15. KMeans 的优缺点

## 优点

```text
1. 简单直观
2. 速度较快
3. 适合大规模向量聚类
4. 输出结果容易解释
5. 每个样本天然有 cluster_id
```

---

## 缺点

```text
1. 需要提前指定 K
2. 对初始中心敏感
3. 对异常点敏感
4. 更适合球状簇
5. 默认使用欧氏距离
6. 不一定适合复杂非凸形状的数据
```

---

# 16. 实际使用建议

对于 10w embedding 聚类，我建议流程是：

```python
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans

# 1. 加载 embedding
X = np.load("embeddings.npy").astype(np.float32)

# 2. 文本 embedding 建议归一化
X = normalize(X, norm="l2")

# 3. 设置聚类数量
num_clusters = 100

# 4. 训练 KMeans
kmeans = MiniBatchKMeans(
    n_clusters=num_clusters,
    batch_size=4096,
    max_iter=100,
    n_init="auto",
    random_state=42
)

labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# 5. 每个 embedding 对应的类簇中心 id
cluster_ids = labels

# 6. 第 i 个 embedding 的类簇中心 id 和中心向量
i = 0
cluster_id = cluster_ids[i]
center_vector = centers[cluster_id]

print(cluster_id)
print(center_vector)
```

---

# 17. 总结

KMeans 的核心处理逻辑是：

```text
初始化 K 个中心
        ↓
每个样本分配给最近中心
        ↓
根据分配结果重新计算中心
        ↓
重复直到收敛
```

对结果的理解：

```python
labels[i]
```

就是：

```text
第 i 个 embedding 对应的类簇中心 id
```

而：

```python
centers[labels[i]]
```

就是：

```text
第 i 个 embedding 对应的类簇中心向量
```

一句话总结：

> **KMeans 就是在不断调整 K 个中心点，让每个 embedding 都归到离自己最近的中心，最后得到每个 embedding 的 cluster_id 和每个 cluster 的中心向量。**

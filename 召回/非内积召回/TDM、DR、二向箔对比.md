- TDM和DR对比：二者很像，都是先预估用户对一个索引结构，然后倒排得到item。
- TDM的索引结构是一个层次化的树，DR的索引结构是一个层次化的路径（二维数组）。 DR索引结构更灵活，表达能力更强。

---
## 总览：这些方法本质上在突破什么

传统 **u2i 双塔向量内积召回** 的核心形式是：

\[
score(u,i)=\langle f_u(u), f_i(i) \rangle
\]

它的优势是 **可离线建 item 向量索引、在线 ANN/MIPS 高效检索**；但天花板也很明显：

1. **表达能力受限**：用户侧和物品侧基本在打分前独立编码，难以表达复杂交叉特征。
2. **单向量难以覆盖多兴趣**：一个 user embedding 容易把多个兴趣平均掉。
3. **内积打分函数低秩约束强**：本质上在用低维向量近似巨大的 user-item 偏好矩阵。
4. **召回候选受 ANN 空间结构限制**：更复杂的深度匹配模型难以直接用于全库召回。

TDM、Deep Retrieval、二向箔类方法，都是围绕一个问题展开：

> **如何在不全库打分的前提下，让召回阶段使用比“单个用户向量 × 单个物品向量内积”更强的匹配函数。**

下面从 **算法原理、训练步骤、在线服务、效果特点、优缺点** 分别展开。

---

# 1. TDM：Tree-based Deep Model

## 1.1 核心思想

**TDM** 将全量 item 组织成一棵树，叶子节点是具体 item，内部节点代表一组 item 的语义聚类。

召回时不再对所有 item 打分，而是从根节点开始逐层向下搜索：

\[
root \rightarrow node_1 \rightarrow node_2 \rightarrow ... \rightarrow item
\]

模型学习的是：

\[
P(node \mid user)
\]

也就是 **用户对某个树节点，即一批 item 集合的兴趣程度**。

这样可以把全库检索从：

\[
O(N)
\]

降低为近似：

\[
O(B \cdot K \cdot H)
\]

其中：

- \(N\)：全量 item 数量
- \(K\)：每个节点的子节点数
- \(H\)：树高，约为 \(\log_K N\)
- \(B\)：beam search 保留的候选节点数

---

## 1.2 算法结构

TDM 主要由两部分组成：

### 1）树结构

树的叶子节点是 item，内部节点是 item cluster。

例如：

```text
Root
├── Node A：手机数码类兴趣
│   ├── Node A1：iPhone 相关
│   │   ├── item_1
│   │   └── item_2
│   └── Node A2：安卓手机相关
├── Node B：母婴类兴趣
└── Node C：食品生鲜类兴趣
```

内部节点的 embedding 通常由其子节点或叶子 item 聚合得到，也可以直接训练。

### 2）用户-节点匹配模型

模型输入通常包括：

- 用户画像特征
- 用户历史行为序列
- 当前上下文特征
- 节点 embedding
- 节点统计特征，如流行度、类目、点击率等

输出：

\[
score(u,n)
\]

表示用户 \(u\) 对节点 \(n\) 的兴趣。

这个打分模型可以是 DNN，不一定是简单内积，因此表达能力比双塔更强。

---

## 1.3 训练流程

### Step 1：构建初始树

常见方式包括：

1. **类目树初始化**
   - 用商品类目、品牌、属性构建树。
   - 优点是可解释。
   - 缺点是类目结构不一定等于用户兴趣结构。

2. **基于 item embedding 聚类**
   - 先训练 item embedding。
   - 用 KMeans 或层次聚类构建树。
   - 更贴近行为相似性。

3. **随机初始化 + 迭代优化**
   - 初始树较粗糙。
   - 后续通过训练反馈调整树结构。

---

### Step 2：构造训练样本

如果用户点击了某个 item，那么该 item 到根节点路径上的所有节点都可以作为正样本。

例如用户点击 item_123：

```text
Root → Node A → Node A1 → item_123
```

则正样本为：

```text
(user, Root)
(user, Node A)
(user, Node A1)
(user, item_123)
```

每一层再采样负节点，例如：

```text
(user, Node A) 是正样本
(user, Node B) 是负样本
(user, Node C) 是负样本
```

训练目标通常是二分类或 softmax：

\[
\mathcal{L} = - y \log p(u,n) - (1-y)\log(1-p(u,n))
\]

或者按层做多分类。

---

### Step 3：训练用户-节点匹配模型

模型学习：

\[
score(u,n)=DNN(user\ features, node\ features)
\]

这里相比双塔召回，TDM 可以引入更多交叉特征，例如：

- 用户最近点击类目与节点类目的交叉
- 用户价格偏好与节点价格带的交叉
- 用户地理位置与节点商品供给区域的交叉
- 用户近期行为序列与节点语义的 attention 匹配

这也是 TDM 相比内积双塔的核心优势之一。

---

### Step 4：树结构迭代优化

树结构对效果影响很大。

常见做法是交替优化：

1. 固定树结构，训练模型。
2. 固定模型，重新调整 item 在树中的位置，使得用户点击 item 的路径得分更高。
3. 重复迭代。

目标是让树结构更符合用户兴趣分布，而不仅仅是商品类目结构。

---

## 1.4 在线召回流程

在线过程是 **Top-down Beam Search**。

### 示例流程

假设每个节点有 \(K=100\) 个子节点，beam size 为 \(B=50\)。

1. 从 root 开始。
2. 对 root 的所有子节点打分。
3. 取 Top-B 个节点。
4. 对这 B 个节点的所有子节点继续打分。
5. 重复直到叶子层。
6. 得到 Top item 作为召回结果。

伪代码：

```python
beam = [root]

for level in range(tree_height):
    candidates = []
    for node in beam:
        children = get_children(node)
        for child in children:
            score = model(user_features, child_features)
            candidates.append((child, score))
    beam = topk(candidates, B)

items = collect_leaf_items(beam)
return topk(items, K)
```

---

## 1.5 在线效率

TDM 的在线复杂度大致为：

\[
O(B \cdot K \cdot H)
\]

例如：

- 全库 item：1 亿
- 树分叉数 \(K=100\)
- 树高 \(H=4\)
- beam size \(B=50\)

理论上在线打分量约为：

\[
50 \times 100 \times 4 = 20,000
\]

相比全库 1 亿 item 打分，大幅降低。

### 效率特点

| 维度 | 表现 |
|---|---|
| 检索复杂度 | 近似 \(O(BKH)\)，远低于全库扫描 |
| 模型复杂度 | 可使用较复杂 DNN，但不能过重 |
| 延迟 | 通常可控，但比纯 ANN 双塔更高 |
| 工程难度 | 中高，需要维护树结构、节点特征、beam search |
| GPU/CPU 适配 | 适合批量打分，可用 GPU 或高性能 CPU 推理 |

---

## 1.6 在线效果特点

TDM 通常在以下方面有优势：

1. **提升召回相关性**
   - 因为使用了更强的 user-node 深度匹配模型。

2. **改善长尾召回**
   - 如果树结构设计合理，长尾 item 可借助父节点获得曝光机会。

3. **提升多兴趣覆盖**
   - Beam search 可同时保留多个兴趣方向。

4. **业务可控性较强**
   - 可以在树节点层面加入类目、价格带、供给策略等约束。

但也存在问题：

1. **路径错误会累积**
   - 如果高层节点没被选中，其子树所有 item 都无法被召回。

2. **树结构质量决定上限**
   - 树构得不好，模型再强也难补救。

3. **训练和线上一致性复杂**
   - 树更新、节点 embedding、样本路径都要保持一致。

---

## 1.7 TDM 优缺点总结

| 方面 | 优点 | 缺点 |
|---|---|---|
| 表达能力 | 可使用复杂 DNN 匹配 user-node | 仍受树结构限制 |
| 效率 | 层次化检索，避免全库打分 | 比 ANN 内积召回更重 |
| 多兴趣 | Beam search 可保留多个方向 | 高层误判会丢失整棵子树 |
| 可解释性 | 树节点可解释，方便运营干预 | 学习型树可解释性下降 |
| 工程实现 | 适合大规模工业系统 | 树构建、更新、部署复杂 |
| 适用场景 | 电商、内容、广告大规模召回 | 对实时兴趣极强的场景需额外优化 |

---

# 2. Deep Retrieval

## 2.1 核心思想

**Deep Retrieval** 可以看作是对 TDM 的进一步泛化。

TDM 依赖一棵树，而 Deep Retrieval 的核心是：

> **学习一个可检索的离散结构，让 item 被映射到若干层离散 code，用户模型预测这些 code，从而完成召回。**

每个 item 不再只用一个连续向量表示，而是被分配一个语义路径或离散编码：

\[
item \rightarrow (c_1, c_2, ..., c_L)
\]

其中：

- \(L\)：编码层数
- \(c_l\)：第 \(l\) 层的离散 code
- 每层 code 数量可以是 \(K\)

例如：

```text
item_123 → [12, 87, 5, 43]
item_456 → [12, 87, 9, 21]
item_789 → [3, 44, 16, 7]
```

用户模型学习：

\[
P(c_1, c_2, ..., c_L \mid user)
\]

并通过 beam search 找到用户最可能感兴趣的 code path，再映射到具体 item。

---

## 2.2 与 TDM 的区别

| 维度 | TDM | Deep Retrieval |
|---|---|---|
| 检索结构 | 显式树结构 | 学习得到的离散 code/path |
| item 表示 | 树中叶子节点 | 多层离散编码 |
| 训练方式 | 通常先建树再训练，可迭代优化 | 更强调结构和模型端到端联合优化 |
| 匹配对象 | user-node | user-code/path |
| 灵活性 | 受树结构约束较强 | code 可学习，结构更灵活 |
| 工程复杂度 | 高 | 更高 |

---

## 2.3 算法原理

Deep Retrieval 将 item 检索建模为一个多层分类问题。

传统召回是：

\[
\arg\max_i score(u,i)
\]

Deep Retrieval 改写为：

\[
\arg\max_{c_1,...,c_L} P(c_1,...,c_L \mid u)
\]

通常可分解为：

\[
P(c_1,...,c_L \mid u)
=
P(c_1 \mid u)
\cdot
P(c_2 \mid u,c_1)
\cdot
...
\cdot
P(c_L \mid u,c_1,...,c_{L-1})
\]

这类似一个层次化生成过程。

每个 item 对应一个或多个 code path，召回时预测高概率 code path，再通过倒排表拿到 item。

---

## 2.4 模型结构

典型 Deep Retrieval 模型包括：

### 1）用户编码器

输入：

- 用户长期画像
- 用户短期行为序列
- 搜索词、场景、时间、地域等上下文
- 设备、渠道、会员状态等特征

输出用户表示：

\[
h_u
\]

用户编码器可以使用：

- DNN
- DIN/DIEN
- Transformer
- MIND/多兴趣网络
- 序列模型

---

### 2）Code 预测层

对于每一层 code，模型预测：

\[
P(c_l \mid u, c_{<l})
\]

实现方式包括：

1. **逐层 softmax**
   - 每层预测一个 code。
   - 简单直接。

2. **条件 softmax**
   - 当前层预测依赖前缀 code。
   - 表达能力更强。

3. **Beam Search 解码**
   - 在线保留 Top-B 条 path。
   - 避免枚举所有 code 组合。

---

### 3）Code-to-Item 倒排表

离线维护：

```text
[12, 87, 5, 43] → item_123, item_982, item_305
[12, 87, 9, 21] → item_456, item_622
```

在线拿到 top code path 后，从倒排表中取出对应 item。

---

## 2.5 训练流程

### Step 1：初始化 item code

初始化方式有多种：

1. **随机分配**
   - 简单，但收敛慢。

2. **基于 item embedding 聚类**
   - 先训练 item embedding。
   - 多层聚类得到 code。

3. **基于类目或业务属性**
   - 例如类目、品牌、价格带。
   - 可解释性强，但不一定最优。

4. **基于行为共现图**
   - 点击、购买、加购共现。
   - 更贴近用户兴趣。

---

### Step 2：训练用户到 code 的预测模型

如果用户点击了 item，而 item 对应 code 为：

```text
item_123 → [12, 87, 5, 43]
```

则训练模型预测该 code path。

损失函数可写为：

\[
\mathcal{L}
=
-\sum_{l=1}^{L}
\log P(c_l \mid u,c_{<l})
\]

---

### Step 3：重新分配 item code

为了让 code 更适合召回，Deep Retrieval 通常会迭代更新 item-code 分配。

目标是让用户真实交互过的 item 的 code path 概率更高：

\[
\max_{code(i)}
\sum_{(u,i)\in D}
\log P(code(i) \mid u)
\]

同时还要满足一些约束：

1. **均衡约束**
   - 避免大量 item 被分到同一个 code。
   - 否则某些桶过大，在线召回效率下降。

2. **唯一性约束**
   - 一个 code path 下 item 数不能无限大。
   - 保证检索粒度。

3. **稳定性约束**
   - 避免频繁改变 item code，导致线上索引震荡。

---

### Step 4：模型和索引交替优化

整体流程类似 EM：

```text
初始化 item code
while not converge:
    固定 item code，训练 user → code 模型
    固定模型，重新优化 item code 分配
    重建 code-to-item 倒排索引
```

---

## 2.6 在线召回流程

### Step 1：用户特征实时编码

得到用户状态：

\[
h_u
\]

### Step 2：逐层预测 code

从第一层开始：

```text
Layer 1: 预测 Top-B 个 c1
Layer 2: 对每个 c1 预测 Top-B 个 c2
Layer 3: 对每个前缀预测 Top-B 个 c3
...
```

### Step 3：Beam Search 保留高分 path

路径得分为：

\[
score(path)
=
\sum_{l=1}^{L}
\log P(c_l \mid u,c_{<l})
\]

### Step 4：倒排取 item

```text
top path → item list
```

### Step 5：轻量排序或规则过滤

对召回 item 做：

- 去重
- 库存过滤
- 类目过滤
- 黑白名单
- 新品/长尾策略
- 粗排重打分

---

## 2.7 在线效率

Deep Retrieval 的在线复杂度主要由两部分组成：

\[
O(B \cdot K \cdot L) + O(M)
\]

其中：

- \(B\)：beam size
- \(K\)：每层候选 code 数
- \(L\)：code 层数
- \(M\)：从 top code path 中取出的 item 数

相比双塔 ANN：

| 维度 | Deep Retrieval | 双塔 ANN |
|---|---|---|
| 检索对象 | code/path | item embedding |
| 复杂度 | 与 code 层数和 beam 有关 | 与 ANN 索引有关 |
| 模型表达 | 更强，可建模复杂路径 | 受内积限制 |
| 在线延迟 | 通常高于 ANN | 通常最低 |
| 索引更新 | code-to-item 倒排更新复杂 | item 向量索引更新成熟 |

---

## 2.8 在线效果特点

Deep Retrieval 的效果优势主要来自：

1. **更强的语义划分**
   - item code 是学习出来的，不完全依赖人工类目。

2. **突破内积低秩限制**
   - 用户到 code path 的预测不是简单向量内积。

3. **天然支持多兴趣**
   - Beam search 可同时保留多个 path。

4. **召回覆盖更灵活**
   - 一个用户可以命中多个语义桶。

5. **适合超大规模 item 集合**
   - 当 item 数极大时，code 检索比 item 级别检索更可控。

但缺点也明显：

1. **训练复杂**
   - code 分配和模型训练相互影响。

2. **线上稳定性挑战大**
   - item code 改变会影响索引、监控和可解释性。

3. **路径错误问题**
   - 高层 code 预测错误会导致后续 item 无法召回。

4. **冷启动依赖额外信息**
   - 新 item 没有行为时，code 分配质量依赖内容特征或类目特征。

---

## 2.9 Deep Retrieval 优缺点总结

| 方面 | 优点 | 缺点 |
|---|---|---|
| 表达能力 | 比双塔内积强，可学习复杂 user-code 关系 | code 结构仍是瓶颈 |
| 检索效率 | 不扫全库，复杂度可控 | beam search 和多层 softmax 有成本 |
| 多兴趣 | 多 path 天然支持多兴趣 | beam 太小会截断兴趣 |
| 训练 | 可联合优化结构和模型 | 训练流程复杂，容易不稳定 |
| 工程 | 适合超大规模召回 | code 分配、索引更新、监控复杂 |
| 效果 | 相关性、覆盖、多兴趣通常更好 | 对冷启动、实时兴趣需额外机制 |

---

# 3. 二向箔类召回：二维化 / 多触发源倒排召回

## 3.1 先明确概念

“二向箔”不是一个像 TDM、Deep Retrieval 那样高度标准化的学术名称。在工业推荐里，通常用它指一类思想：

> **把复杂的 user → item 召回问题，拆成多个可离线预计算的二元关系，再在线快速组合。**

典型形式是：

\[
user \rightarrow trigger \rightarrow item
\]

其中 trigger 可以是：

- 用户最近点击的商品
- 用户购买过的商品
- 用户搜索 query
- 用户关注的店铺
- 用户偏好的类目
- 用户感兴趣的品牌
- 用户所在人群分桶
- 用户短期 session 意图

然后离线预计算：

\[
score(trigger, item)
\]

在线根据用户的多个 trigger 拉取 item，再聚合。

---

## 3.2 为什么它能突破双塔内积

双塔召回是：

\[
score(u,i)=\langle f_u(u), f_i(i) \rangle
\]

二向箔类方法通常是：

\[
score(u,i)=\sum_{t \in T(u)} w(u,t) \cdot score(t,i)
\]

其中：

- \(T(u)\)：用户的 trigger 集合
- \(w(u,t)\)：用户对 trigger 的权重
- \(score(t,i)\)：trigger 到 item 的离线关系分

这个形式本质上是 **稀疏多兴趣匹配**。

它不强迫用户所有兴趣压缩成一个 dense vector，而是保留多个显式兴趣触发点。

例如用户最近行为：

```text
用户最近点击：
1. iPhone 15 手机壳
2. 露营帐篷
3. 猫粮
```

双塔可能把这些兴趣压到一个向量里，出现兴趣混合。

二向箔则可以分别召回：

```text
iPhone 15 手机壳 → 钢化膜、充电器、MagSafe 配件
露营帐篷 → 睡袋、防潮垫、露营灯
猫粮 → 猫砂、猫罐头、宠物玩具
```

再进行加权融合。

---

## 3.3 算法原理

二向箔类方法的关键是构造一个二维关系表：

\[
trigger \times item
\]

例如：

```text
trigger_item_id → top related item_ids
query → top item_ids
category → top item_ids
brand → top item_ids
user_cluster → top item_ids
```

每个 trigger 对应一个 item 候选列表。

离线可以使用复杂模型计算：

\[
score(t,i)
\]

因为这是离线计算，不受在线毫秒级延迟限制，可以使用更重的模型。

---

## 3.4 离线构建流程

### Step 1：定义 trigger 类型

常见 trigger 类型包括：

| Trigger 类型 | 示例 | 适合场景 |
|---|---|---|
| item trigger | 用户点击过的商品 | i2i、搭配、替代品召回 |
| query trigger | 用户搜索词 | 搜推广联动、搜索后推荐 |
| category trigger | 用户偏好类目 | 中长期兴趣召回 |
| brand trigger | 用户偏好品牌 | 品牌忠诚度强的电商场景 |
| shop trigger | 用户关注店铺 | 店铺粉丝推荐 |
| author/content trigger | 内容作者、视频主题 | 内容推荐 |
| user cluster trigger | 用户人群分桶 | 冷启动、粗粒度召回 |

---

### Step 2：构造 trigger-item 训练样本

以 item trigger 为例。

如果用户行为序列是：

```text
A → B → C → D
```

可以构造：

```text
A → B
A → C
B → C
B → D
C → D
```

正样本来自：

- 同 session 点击
- 点击后购买
- 加购后购买
- 搜索后点击
- 点击后停留
- 商品搭配购买
- 短时间共现

负样本可以来自：

- 曝光未点击
- 同类目未点击
- 热门 item 采样
- 随机采样
- hard negative

---

### Step 3：训练 trigger-item 打分模型

可以有多种建模方式。

#### 方式一：共现统计

常用指标：

\[
sim(i,j)=\frac{cooccur(i,j)}{\sqrt{freq(i)freq(j)}}
\]

或：

\[
sim(i,j)=\frac{cooccur(i,j)}{freq(i)^\alpha freq(j)^\beta}
\]

优点：

- 简单
- 高效
- 可解释
- 实时更新容易

缺点：

- 表达能力有限
- 容易偏热门
- 冷启动弱

---

#### 方式二：ItemCF / Swing

在电商场景中，Swing 类方法常用于 i2i 召回。

它强调两个用户共同点击两个 item 的可靠性。

直觉是：

- 如果很多用户都点击了 A 和 B，A 与 B 相关。
- 但如果这些用户本身点击很多商品，贡献应降低。
- 如果 A、B 只被少数高重合用户连接，关系更可信。

简化形式：

\[
sim(i,j)=\sum_{u,v \in U(i)\cap U(j)} \frac{1}{\alpha + |I(u)\cap I(v)|}
\]

这类方法对电商推荐中的 **相似品、替代品、搭配品** 比较有效。

---

#### 方式三：Graph Embedding / GNN

构建异构图：

```text
user - item
item - category
item - brand
item - shop
query - item
```

然后学习 trigger 和 item 的关系。

可用方法包括：

- random walk
- node2vec
- PinSage 类图召回
- GraphSAGE
- LightGCN
- heterogeneous GNN

优点：

- 能融合多种关系。
- 对长尾 item 友好。
- 可以发现多跳兴趣。

缺点：

- 图构建和增量更新复杂。
- 在线解释性较弱。

---

#### 方式四：离线重模型打分

对于每个 trigger，先通过粗召回拿到一批 item，再用复杂模型重排：

\[
score(t,i)=DNN(t,i,context)
\]

模型可以引入丰富交叉特征：

- trigger 类目 × item 类目
- trigger 品牌 × item 品牌
- trigger 价格带 × item 价格带
- trigger embedding × item embedding
- 用户转移行为
- 购买间隔
- 复购周期
- 搭配关系
- 替代关系

最终为每个 trigger 保留 Top-M item。

---

### Step 4：建立倒排索引

离线输出：

```text
trigger_1 → [(item_1, score), (item_2, score), ...]
trigger_2 → [(item_7, score), (item_9, score), ...]
```

通常每个 trigger 保留 Top 100、Top 500 或 Top 1000，视业务规模而定。

---

## 3.5 在线召回流程

### Step 1：生成用户 trigger

从用户行为中抽取 trigger：

```text
最近点击 item
最近购买 item
最近搜索 query
高频浏览类目
偏好品牌
关注店铺
```

对每个 trigger 计算权重：

\[
w(u,t)
\]

权重通常考虑：

- 行为类型：购买 > 加购 > 点击 > 曝光
- 时间衰减：越近权重越高
- 行为强度：停留、点击次数、购买金额
- 场景匹配：当前频道、当前页面、当前 query

---

### Step 2：查倒排表

对每个 trigger 拉取 Top-M item：

```text
for trigger in user_triggers:
    candidates += index[trigger][:M]
```

---

### Step 3：多路融合

同一个 item 可能被多个 trigger 召回，需要聚合：

\[
score(u,i)=\sum_{t \in T(u)} w(u,t)\cdot score(t,i)
\]

也可以加入：

- 最大分
- 平均分
- trigger 数量
- trigger 类型权重
- 时间衰减
- 类目多样性惩罚
- 热度惩罚

---

### Step 4：过滤和截断

常见规则：

- 去重
- 已购过滤
- 库存过滤
- 低质商品过滤
- 价格带过滤
- 类目黑名单
- 曝光疲劳控制
- 多样性打散

---

## 3.6 在线效率

二向箔类召回在线通常非常高效。

假设：

- 用户 trigger 数：\(T=20\)
- 每个 trigger 拉取：\(M=500\)

则最多拉取：

\[
20 \times 500 = 10,000
\]

个候选，然后去重、融合、截断。

复杂度约为：

\[
O(T \cdot M)
\]

主要成本是：

- KV 查询
- 候选列表 merge
- 去重
- 轻量打分

一般比 TDM 和 Deep Retrieval 更容易做到低延迟。

---

## 3.7 在线效果特点

二向箔类方法通常在以下场景效果很好：

1. **短期兴趣强**
   - 用户刚看过某商品，立刻推荐相关商品。

2. **电商 i2i 关系强**
   - 替代品、搭配品、同品牌、同价位商品推荐。

3. **用户兴趣多样**
   - 多个 trigger 可以分别召回不同兴趣方向。

4. **工程稳定性要求高**
   - 倒排表简单可靠，易监控。

5. **解释性要求强**
   - 可以解释为“因为你看过 A，所以推荐 B”。

但缺点也明显：

1. **容易过度相似**
   - 推荐结果可能围绕近期行为打转，探索性不足。

2. **依赖用户历史行为**
   - 新用户 trigger 少，效果弱。

3. **冷启动 item 难进入倒排表**
   - 没有共现或行为的新商品难被召回。

4. **离线表膨胀**
   - trigger 数量大时，存储压力明显。

5. **全局最优能力有限**
   - 它更像局部相关性扩展，不一定能捕获复杂全局偏好。

---

## 3.8 二向箔类方法优缺点总结

| 方面 | 优点 | 缺点 |
|---|---|---|
| 表达能力 | 保留多个显式兴趣 trigger，避免单向量平均 | trigger-item 关系仍偏局部 |
| 效率 | KV 查询 + merge，在线极快 | 候选过多时 merge 成本上升 |
| 短期兴趣 | 很强，适合实时兴趣 | 容易兴趣窄化 |
| 可解释性 | 强，可解释为相似、搭配、同类 | 复杂图模型解释性下降 |
| 工程实现 | 简单稳定，易灰度 | 离线表维护和存储压力大 |
| 冷启动 | 对新用户、新 item 较弱 | 需要内容召回或探索流量补充 |
| 业务控制 | 方便做类目、品牌、搭配策略 | 策略过多会影响模型一致性 |

---

# 4. 三类方法横向对比

## 4.1 核心机制对比

| 方法 | 核心思想 | 检索结构 | 在线检索方式 | 是否突破内积 |
|---|---|---|---|---|
| 双塔 ANN | user/item 向量内积 | 向量索引 | ANN/MIPS | 否 |
| TDM | item 组织成树，逐层找兴趣节点 | 树 | Beam Search | 是 |
| Deep Retrieval | 学习 item 离散 code/path | 多层 code 倒排 | Code Beam Search | 是 |
| 二向箔 | user → trigger → item | trigger-item 倒排 | 多 trigger 查表融合 | 是 |

---

## 4.2 表达能力对比

| 方法 | 表达能力 | 说明 |
|---|---|---|
| 双塔 ANN | 中 | 受内积和向量维度限制 |
| TDM | 高 | user-node 可用 DNN 建模交叉 |
| Deep Retrieval | 高 | user-code/path 可端到端学习 |
| 二向箔 | 中高 | 多 trigger 保留多兴趣，但局部关系较强 |

表达能力大致可以理解为：

```text
Deep Retrieval ≈ TDM > 二向箔 > 双塔内积
```

但这不是绝对的。二向箔如果离线用重模型构建 trigger-item 表，也可以非常强。

---

## 4.3 在线效率对比

| 方法 | 在线复杂度 | 延迟特点 |
|---|---|---|
| 双塔 ANN | \(O(\log N)\) 或近似 ANN | 最快、最成熟 |
| 二向箔 | \(O(T \cdot M)\) | 很快，主要是 KV + merge |
| TDM | \(O(B \cdot K \cdot H)\) | 可控，但 DNN 打分较多 |
| Deep Retrieval | \(O(B \cdot K \cdot L)\) | 可控，但 beam 和多层预测较重 |

在线效率大致为：

```text
双塔 ANN ≥ 二向箔 > TDM ≈ Deep Retrieval
```

如果 TDM/Deep Retrieval 使用轻量模型并做批量推理，延迟也可以控制得很好。

---

## 4.4 效果特点对比

| 方法 | 相关性 | 多兴趣 | 长尾 | 冷启动 | 可解释性 |
|---|---|---|---|---|---|
| 双塔 ANN | 中 | 弱到中 | 中 | 依赖内容特征 | 弱 |
| TDM | 高 | 中高 | 中高 | 中 | 中 |
| Deep Retrieval | 高 | 高 | 中高 | 中 | 中低 |
| 二向箔 | 中高 | 高 | 中 | 弱到中 | 高 |

---

## 4.5 工程复杂度对比

| 方法 | 工程复杂度 | 主要难点 |
|---|---|---|
| 双塔 ANN | 中 | 向量索引、增量更新、ANN 参数 |
| 二向箔 | 中 | 倒排表规模、trigger 管理、融合策略 |
| TDM | 高 | 树构建、树更新、beam search、训练线上一致 |
| Deep Retrieval | 很高 | code 学习、索引稳定性、训练收敛、在线解码 |

工程复杂度大致为：

```text
Deep Retrieval > TDM > 二向箔 ≈ 双塔 ANN
```

---

# 5. 更细的优缺点对比

## 5.1 TDM vs Deep Retrieval

| 维度 | TDM | Deep Retrieval |
|---|---|---|
| 结构 | 显式树 | 学习型离散编码 |
| 可解释性 | 较好 | 较弱 |
| 灵活性 | 中 | 高 |
| 训练难度 | 高 | 更高 |
| 在线稳定性 | 较好 | code 更新可能导致震荡 |
| 效果上限 | 高 | 更高，但依赖训练质量 |
| 适合场景 | 类目层次明显的大规模推荐 | 超大规模、语义复杂、希望端到端学习索引的场景 |

### 结论

- 如果业务有天然类目树、希望可控和可解释，**TDM 更稳**。
- 如果追求更高表达能力，且有较强工程和训练能力，**Deep Retrieval 上限更高**。

---

## 5.2 TDM vs 二向箔

| 维度 | TDM | 二向箔 |
|---|---|---|
| 召回路径 | root 到 item | user 到 trigger 到 item |
| 短期兴趣 | 中高 | 很强 |
| 多兴趣 | Beam 保留多路径 | 多 trigger 天然支持 |
| 在线效率 | 需要多次模型打分 | KV 查表，通常更快 |
| 工程稳定性 | 较复杂 | 较稳定 |
| 可解释性 | 中 | 高 |
| 长尾能力 | 依赖树结构 | 依赖 trigger-item 表 |
| 冷启动 | 中 | 弱到中 |

### 结论

- **TDM 更像全局语义检索**。
- **二向箔更像局部兴趣扩展**。
- 电商场景中，二者常常不是替代关系，而是互补关系。

---

## 5.3 Deep Retrieval vs 二向箔

| 维度 | Deep Retrieval | 二向箔 |
|---|---|---|
| 模型化程度 | 高 | 中 |
| 结构学习 | item code 可学习 | trigger-item 表多为离线统计/模型打分 |
| 在线速度 | 中高 | 高 |
| 短期行为利用 | 依赖用户编码器 | 很直接 |
| 可解释性 | 较弱 | 强 |
| 工程难度 | 很高 | 中 |
| 效果上限 | 高 | 中高 |
| 稳定性 | 需重点保障 | 通常较稳 |

### 结论

- Deep Retrieval 适合做 **主力深度召回通道**。
- 二向箔适合做 **高精度、多兴趣、强解释的补充召回通道**。

---

# 6. 适用场景建议

## 6.1 电商首页推荐

建议组合：

```text
双塔 ANN + 二向箔 + TDM/Deep Retrieval
```

原因：

- 双塔 ANN：保证大规模泛化和基础覆盖。
- 二向箔：强化最近兴趣和商品相关性。
- TDM/Deep Retrieval：提升复杂兴趣建模和长尾覆盖。

---

## 6.2 商品详情页相关推荐

优先：

```text
二向箔 / i2i / Swing / Graph Recall
```

原因：

- 当前商品就是强 trigger。
- 用户意图明确。
- 需要相似品、搭配品、替代品。

TDM 和 Deep Retrieval 可以作为补充，但不是最自然的主通道。

---

## 6.3 信息流推荐

优先：

```text
双塔 ANN + Deep Retrieval + 多兴趣召回
```

原因：

- 内容更新快。
- 用户兴趣多样。
- item 语义复杂。
- 需要更强的序列兴趣建模。

二向箔可用于：

- 作者相似
- topic 相似
- 最近点击内容扩展

---

## 6.4 搜索推荐 / 搜推联动

适合：

```text
query-trigger 二向箔 + Deep Retrieval
```

原因：

- query 是非常强的 intent trigger。
- 可以离线构建 query-item 倒排表。
- Deep Retrieval 可以补充语义泛化能力。

---

# 7. 实际落地时的关键指标

评估这些召回方法时，不建议只看 Recall@K，还应综合看：

## 7.1 离线指标

| 指标 | 含义 |
|---|---|
| Recall@K | 用户真实点击/购买是否被召回 |
| HitRate@K | 是否至少命中一个正样本 |
| NDCG@K | 召回排序质量 |
| Coverage | item 覆盖率 |
| Long-tail Coverage | 长尾商品覆盖 |
| Diversity | 类目、品牌、店铺多样性 |
| Novelty | 新颖性 |
| Popularity Bias | 热门偏置程度 |

---

## 7.2 在线指标

| 指标 | 含义 |
|---|---|
| CTR | 点击率 |
| CVR | 转化率 |
| GMV | 成交额 |
| RPM | 千次曝光收益 |
| Add-to-cart Rate | 加购率 |
| Order Rate | 下单率 |
| Latency | 召回耗时 |
| Timeout Rate | 超时率 |
| Candidate Contribution | 最终排序页中来自该通道的占比 |
| Unique Item Coverage | 实际曝光商品覆盖 |
| User Coverage | 覆盖用户比例 |

---

# 8. 总结建议

## 8.1 方法选择

| 业务诉求 | 推荐方法 |
|---|---|
| 极致低延迟、成熟稳定 | 双塔 ANN、二向箔 |
| 强短期兴趣、相似品/搭配品 | 二向箔 |
| 大规模全局语义召回 | TDM |
| 希望突破双塔上限、学习检索结构 | Deep Retrieval |
| 多兴趣覆盖 | 二向箔、Deep Retrieval、TDM Beam Search |
| 可解释和运营可控 | 二向箔、TDM |
| 追求模型上限 | Deep Retrieval、TDM |

---

## 8.2 推荐的工业级召回架构

更稳妥的方式不是押注单一方法，而是多通道融合：

```text
召回层：
1. 双塔 ANN：基础泛化召回
2. 二向箔/i2i：短期兴趣和商品关系召回
3. TDM：树结构语义召回
4. Deep Retrieval：学习型离散结构召回
5. 热门/新品/探索召回：补充覆盖

融合层：
- 通道配额
- 分数归一化
- 去重
- 多样性控制
- 业务规则过滤

粗排层：
- 统一模型重打分

精排层：
- 多目标排序
```

---

## 8.3 一句话总结

- **TDM**：用树把全库 item 分层组织，在线逐层搜索，兼顾表达能力和效率。
- **Deep Retrieval**：学习 item 的离散语义 code，让召回变成 code path 预测，上限高但工程复杂。
- **二向箔**：把 user→item 拆成 user→trigger→item，多 trigger 倒排召回，在线快、解释强、短期兴趣效果好。
- 三者都在突破双塔内积限制，但侧重点不同：  
  **TDM 偏层次语义检索，Deep Retrieval 偏端到端学习索引，二向箔偏多兴趣触发和局部关系扩展。**

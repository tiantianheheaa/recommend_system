 在生成式推荐（Generative Recommendation）中，RQ-VAE（Residual Quantized Variational AutoEncoder）负责将连续的 Item 语义嵌入（如 Sentence-T5 编码的标题、类别、价格等）压缩成一组离散的 **Semantic ID**（如 $(c_1, c_2, c_3)$），供下游的序列生成模型（如 Transformer）进行自回归预测。其训练目标由 **三个核心 Loss** 组成，下面给出标准公式、逐层扩展形式以及各自的作用与含义。

---

## 一、总体损失函数

RQ-VAE 采用多层残差量化（深度为 $L$，每层码本大小为 $K$）。设输入为 Item 语义嵌入 $\mathbf{x} \in \mathbb{R}^D$，编码器输出为 $\mathbf{z}_e^{(1)} = \text{Encoder}(\mathbf{x})$，第 $l$ 层的量化残差为 $\mathbf{r}_l$，选中的码本向量为 $\mathbf{e}_{c_l}^{(l)} \in \mathbb{R}^d$（$c_l \in \{1,\dots,K\}$ 为离散码），$sg[\cdot]$ 为 stop-gradient 算子。则总损失为：

$$
\mathcal{L}_{\text{RQ-VAE}} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|^2}_{\text{Reconstruction Loss}} + \sum_{l=1}^{L} \Big( \underbrace{\|sg[\mathbf{r}_l] - \mathbf{e}_{c_l}^{(l)}\|^2}_{\text{Codebook Loss}} + \beta \underbrace{\|\mathbf{r}_l - sg[\mathbf{e}_{c_l}^{(l)}]\|^2}_{\text{Commitment Loss}} \Big)
$$

其中 $\hat{\mathbf{x}} = \text{Decoder}\big(\sum_{l=1}^{L} \mathbf{e}_{c_l}^{(l)}\big)$ 为解码器重构输出，$\beta > 0$（通常取 $0.25 \sim 1.0$）为平衡系数。

---

## 二、三个 Loss 的详细说明

### 1. Reconstruction Loss（重建损失）

$$
\mathcal{L}_{\text{rec}} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \Big\|\mathbf{x} - \text{Decoder}\Big(\sum_{l=1}^{L} \mathbf{e}_{c_l}^{(l)}\Big)\Big\|^2
$$

**作用**：  
这是 AutoEncoder 的根基，要求解码器能够从多层量化码字的**累加和**中尽可能精确地恢复原始 Item 嵌入 $\mathbf{x}$。在推荐场景中，它直接决定了 Semantic ID 对 Item 语义信息的保留程度——重建误差越小，说明离散码 $(c_1, \dots, c_L)$ 对原始 Item 的语义刻画越精准。

**含义**：  
- 信息瓶颈：量化过程会丢失信息，重建损失迫使模型在离散约束下最小化信息损失。  
- 语义保真：保证具有相似 Semantic ID 的 Item 在原始嵌入空间中也彼此接近，从而为下游生成模型提供有意义的 Token 序列。

---

### 2. Codebook Loss / Dictionary Loss（码本损失）

$$
\mathcal{L}_{\text{codebook}} = \sum_{l=1}^{L} \|sg[\mathbf{r}_l] - \mathbf{e}_{c_l}^{(l)}\|^2
$$

其中残差递推定义为：
- 第 1 层：$\mathbf{r}_1 = \mathbf{z}_e^{(1)} = \text{Encoder}(\mathbf{x})$
- 第 $l$ 层：$\mathbf{r}_l = \mathbf{r}_{l-1} - \mathbf{e}_{c_{l-1}}^{(l-1)}$（前一层残差减去已选码本向量）

**作用**：  
专门用于**更新码本向量** $\mathbf{e}_{c_l}^{(l)}$。由于 $sg[\cdot]$ 的存在，梯度不会回传到编码器，此 Loss 相当于把编码器输出（或残差）当作固定目标，通过梯度下降将选中的码本向量拉向该目标。

**含义**：  
- 字典学习：类比传统 Vector Quantization，它让码本中的每个码字成为其对应聚类区域的质心。  
- 码本维护：防止码本向量长期不更新（codebook collapse）。在推荐系统中，若码本向量不能紧跟 Item 嵌入分布，会导致大量 Item 被映射到少数几个码，造成 Semantic ID 区分度下降。

---

### 3. Commitment Loss（承诺损失）

$$
\mathcal{L}_{\text{commit}} = \sum_{l=1}^{L} \|\mathbf{r}_l - sg[\mathbf{e}_{c_l}^{(l)}]\|^2
$$

**作用**：  
专门用于**约束编码器输出**（或残差）。此时 $sg[\cdot]$ 施加在码本向量上，梯度只流向编码器，迫使编码器产生的向量主动"承诺"靠近当前码本中的某个码字，而不是无限漂移。

**含义**：  
- 防止体积坍塌（Volume Collapse）：如果没有此 Loss，编码器可能输出任意大的向量，导致量化时总是选中少数几个极端码字，其余码字永不激活（即 codebook collapse）。  
- 稳定训练：在推荐场景中，Item 嵌入空间通常高维且分布复杂，Commitment Loss 确保编码器输出的残差始终落在码本向量的有效邻域内，使多层残差量化能够逐层精细分解信息。  
- 超参数 $\beta$：控制编码器"遵守承诺"的强度。$\beta$ 过小则编码器容易逃逸，过大则限制表达力；在 TIGER 等生成式推荐工作中通常设为 $0.25$。

---

## 三、三者的协同关系

| Loss | 优化对象 | 被约束对象 | 核心目的 |
|------|----------|------------|----------|
| Reconstruction | 编码器 + 解码器 | 整体重构质量 | 保证 Semantic ID 能还原 Item 语义 |
| Codebook | 码本向量 | 码本与残差的距离 | 让码本向量成为好的聚类中心 |
| Commitment | 编码器 | 编码器输出范围 | 防止编码器逃逸，维持码本利用率 |

在训练时，RQ-VAE 通过 Straight-Through Estimator（STE）绕过不可导的 $\arg\min$ 量化操作，使得重建误差能回传至编码器；而 Codebook Loss 和 Commitment Loss 则通过 stop-gradient 明确分工，分别优化码本和编码器。三者共同保证了生成式推荐系统中离散 Semantic ID 的**高保真**、**高区分度**和**训练稳定性**。

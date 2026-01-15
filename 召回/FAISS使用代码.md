这里提供一套完整的 Faiss 使用示例代码，从**基础环境准备**到**四种核心场景**（Flat, IVF, HNSW, IVF-PQ）的实战演示，最后还包括**GPU加速**和**磁盘读写**的进阶用法。

### 0. 环境准备

首先确保安装了 Faiss 和 NumPy：

```bash
# CPU 版本
pip install faiss-cpu numpy

# 如果有 GPU (CUDA 11.0+)，建议安装 GPU 版本，速度提升巨大
# pip install faiss-gpu numpy
```

---

### 1. 基础数据准备

我们在所有示例中使用同一组随机数据，以便对比。

```python
import numpy as np
import faiss

# 设置随机种子以便复现
np.random.seed(42)

# 数据参数
d = 64                           # 向量维度
nb = 10000                       # 数据库向量数量 (1万条)
nq = 5                           # 查询向量数量
k = 10                           # 返回 Top-K 个结果

# 生成随机数据库向量 (必须是 float32)
xb = np.random.random((nb, d)).astype('float32')

# 生成随机查询向量
xq = np.random.random((nq, d)).astype('float32')

# 为了验证结果，我们手动计算前10个真实的最近邻 (仅用于小数据集演示)
# 在大数据集上不要这样做，因为 O(N^2) 太慢了
D_brute, I_brute = faiss.KNN(xb, xq, k, metric=faiss.METRIC_L2)
print("真实最近邻索引 (前5个):", I_brute[:5, :3])
```

---

### 场景一：Flat 索引 (暴力搜索)

**适用**：小规模数据，或作为精度基准。

```python
print("\n=== 1. Flat 索引 (精确搜索) ===")

# 创建索引
index_flat = faiss.IndexFlatL2(d)  # L2 距离 (欧氏距离)
# index_flat = faiss.IndexFlatIP(d) # 如果用内积相似度，用这个

print(f"索引是否训练: {index_flat.is_trained}")

# 添加数据
index_flat.add(xb)
print(f"索引中的向量总数: {index_flat.ntotal}")

# 搜索
D, I = index_flat.search(xq, k)

print("查询结果 (前5个):")
print("距离:\n", D[:5, :3])
print("索引:\n", I[:5, :3])

# 验证召回率 (因为是暴力搜索，应该是 100%)
# 注意：如果有距离相同的情况，排序可能略有不同，这里简单对比
hits = 0
for i in range(nq):
    if set(I[i]) & set(I_brute[i]):
        hits += 1
print(f"召回率 (对比暴力基准): {hits / nq * 100:.2f}% (应为 100%)")
```

---

### 场景二：IVF 索引 (倒排文件)

**适用**：大规模数据，需要在速度和精度间权衡。

```python
print("\n=== 2. IVF 索引 (聚类加速) ===")

nlist = 100  # 聚类中心数量 (关键参数)
quantizer = faiss.IndexFlatL2(d)  # 量化器，通常用 Flat

# 创建 IVF 索引
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# 重要：必须先训练 (Training) 才能添加数据
print("开始训练索引...")
index_ivf.train(xb)
print(f"索引是否训练: {index_ivf.is_trained}")

# 添加数据
index_ivf.add(xb)
print(f"索引中的向量总数: {index_ivf.ntotal}")

# 搜索前需要设置 nprobe (搜索时检查的簇数量)
index_ivf.nprobe = 10  # 默认是 1。越大越准越慢，越小越快越不准

# 搜索
D_ivf, I_ivf = index_ivf.search(xq, k)

print("查询结果 (前5个):")
print("索引:\n", I_ivf[:5, :3])

# 对比 Flat 的结果，看是否一致 (通常大部分一致)
print("与 Flat 索引结果对比 (前5个第1名):", np.array_equal(I[:5, 0], I_ivf[:5, 0]))
```

---

### 场景三：HNSW 索引 (图结构)

**适用**：中大规模数据，对查询延迟敏感，支持动态增删。

```python
print("\n=== 3. HNSW 索引 (图结构) ===")

# 创建 HNSW 索引
# M: 每个节点的连接数 (越大图越密，精度高但内存大)
# efSearch: 搜索深度 (越大越准越慢)
index_hnsw = faiss.IndexHNSWFlat(d, 32)  # M=32 是常用默认值
index_hnsw.hnsw.efSearch = 64  # 搜索时的探索范围

# HNSW 不需要 train，直接 add
index_hnsw.add(xb)
print(f"索引中的向量总数: {index_hnsw.ntotal}")

# 搜索
D_hnsw, I_hnsw = index_hnsw.search(xq, k)

print("查询结果 (前5个):")
print("索引:\n", I_hnsw[:5, :3])

# 验证：HNSW 通常能达到接近 100% 的召回率
hits = 0
for i in range(nq):
    if set(I_hnsw[i]) == set(I_brute[i]): # HNSW 效果通常很好，尝试全匹配
        hits += 1
print(f"完美召回率: {hits / nq * 100:.2f}%")
```

---

### 场景四：IVF + PQ (乘积量化压缩)

**适用**：超大规模数据（亿级），内存极其受限。

```python
print("\n=== 4. IVF-PQ 索引 (压缩内存) ===")

nlist = 100
m = 8  # PQ 分段数 (d 必须能被 m 整除, 64/8=8)
bits = 8 # 每个子向量用 8 bit 表示

quantizer = faiss.IndexFlatL2(d)
# 创建 IVF-PQ 索引
index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)

# 训练和添加
index_ivfpq.train(xb)
index_ivfpq.add(xb)

# 设置 nprobe
index_ivfpq.nprobe = 10

# 搜索
D_pq, I_pq = index_ivfpq.search(xq, k)

print("查询结果 (前5个):")
print("索引:\n", I_pq[:5, :3])

# 检查内存占用 (对比 Flat 和 IVF-PQ)
mem_flat = index_flat.total_storage * 4 / 1024 / 1024 # MB
mem_ivfpq = index_ivfpq.total_storage * 4 / 1024 / 1024 # MB
print(f"\nFlat 索引内存: {mem_flat:.2f} MB")
print(f"IVF-PQ 索引内存: {mem_ivfpq:.2f} MB (压缩了约 {mem_flat/mem_ivfpq:.1f} 倍)")
```

---

### 场景五：进阶技巧 (GPU & 磁盘IO)

#### 5.1 GPU 加速

如果你安装了 `faiss-gpu`，可以将索引转移到 GPU 上，速度提升 10-50 倍。

```python
print("\n=== 5. GPU 加速 ===")

# 检查 GPU 是否可用
if faiss.get_num_gpus() > 0:
    # 创建 CPU 端的 HNSW 索引
    index_cpu = faiss.IndexHNSWFlat(d, 32)
    index_cpu.add(xb)
    
    # 转换为 GPU 索引 (需要指定 GPU ID，这里用 0)
    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    
    print("索引已转移到 GPU")
    
    # 在 GPU 上搜索 (语法完全一样)
    D_gpu, I_gpu = index_gpu.search(xq, k)
    print("GPU 搜索结果 (前5个):\n", I_gpu[:5])
else:
    print("未检测到 GPU，跳过此步骤")
```

#### 5.2 索引的保存与加载

生产环境中，索引构建好后需要持久化到磁盘。

```python
print("\n=== 6. 索引读写 (磁盘持久化) ===")

# 假设我们要保存 HNSW 索引
faiss.write_index(index_hnsw, "hnsw_index.faiss")
print("索引已保存到 hnsw_index.faiss")

# 加载索引
index_loaded = faiss.read_index("hnsw_index.faiss")
print(f"加载的索引向量数: {index_loaded.ntotal}")

# 验证加载后的索引是否能正常搜索
D_load, I_load = index_loaded.search(xq, k)
assert np.array_equal(I_hnsw, I_load)
print("加载后的索引搜索结果与原索引一致")
```

### 总结与参数调优建议

1.  **索引选择逻辑**：
    *   数据 < 10万：`IndexFlatL2`
    *   数据 10万 - 1000万，追求高召回+低延迟：`IndexHNSWFlat`
    *   数据 > 1000万，内存有限：`IndexIVFPQ` (先试 `IndexIVFFlat` 如果内存够)
2.  **关键参数**：
    *   **IVF**: `nprobe` 是核心。从 10 开始试，太小召回低，太大速度慢。
    *   **HNSW**: `efSearch` (查询时) 和 `efConstruction` (构建时)。构建时越大越好（如 200+），查询时 32-64 通常够用。
    *   **PQ**: `nprobe` 依然重要，`m` (分段数) 越大精度越高但内存越大。
3.  **距离度量**：
    *   `METRIC_L2`: 欧氏距离（最常用）。
    *   `METRIC_INNER_PRODUCT`: 内积。如果向量已归一化，内积等价于余弦相似度。使用时需配合 `IndexFlatIP`。

sklearn 中 KMeans 与 MiniBatchKMeans 的区别
一、核心定位
表格
维度	KMeans	MiniBatchKMeans
本质	经典批量 K-Means	K-Means 的‌随机梯度下降（SGD）加速版‌
每次迭代数据	‌全量数据‌	‌小批量（mini-batch）抽样数据‌
适用场景	中小规模数据，追求最优聚类质量	大规模数据（如 >1万样本），追求速度
聚类质量	通常更优	略差，但实践中差异不明显
二、算法流程对比
KMeans（标准版）
初始化 K 个质心（默认 k-means++）
遍历全部样本‌，计算每个样本到各质心的距离，分配到最近簇
基于全量数据‌重新计算每个簇的均值作为新质心
重复 2-3，直到质心移动 < tol 或达到 max_iter
MiniBatchKMeans（小批量版）
初始化 K 个质心（同样支持 k-means++）
从全量数据中‌随机抽取一个小批量（batch_size）‌
将该批次样本分配给最近质心
用‌滑动平均‌更新质心：

𝑐
new
=
(
1
−
𝜂
)
⋅
𝑐
old
+
𝜂
⋅
𝑥
c
new
	​

=(1−η)⋅c
old
	​

+η⋅x

其中 
𝜂
η 为学习率，随该质心历史处理样本数 
𝑛
n 自动衰减（
𝜂
∝
1
/
𝑛
η∝1/n）
5. 重复 2-4，直到收敛

关键区别‌：标准 KMeans 每轮用全量数据取"算术平均"更新质心；MiniBatchKMeans 每轮只用一个小批量做"滑动平均"更新，中心越成熟受新样本影响越小。

三、关键参数差异
表格
参数	KMeans	MiniBatchKMeans	说明
n_clusters	✅	✅	聚类簇数 K
init	✅ 默认 'k-means++'	✅ 默认 'k-means++'	初始化方式
n_init	✅ 默认 'auto'	✅ 默认 'auto'	不同初始化运行次数
max_iter	✅ 默认 300	✅ 默认 100	最大迭代次数
tol	✅ 默认 1e-4	✅ 默认 0.0	收敛阈值
batch_size	❌	✅ 默认 1024	‌MiniBatch 独有‌：每批样本数
algorithm	✅ 'lloyd'/'elkan'	❌ 仅 'lloyd'	KMeans 可选 elkan 加速
四、性能与效果对比
速度
MiniBatchKMeans 快数倍到数十倍‌，因为每次只处理 batch_size 个样本，而非全量
数据量越大，速度优势越明显
质量
标准 KMeans 通常更优（SSE/inertia 更小）
MiniBatchKMeans 质量略差，但在实际项目中‌差异通常不明显‌
收敛性
KMeans：收敛到局部最优（多次 n_init 缓解）
MiniBatchKMeans：由于噪声引入，可能在最优解附近波动，但学习率衰减保证最终稳定
五、代码示例
python
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=10000, centers=4, random_state=42)

# 标准 KMeans
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X)

# MiniBatch KMeans
mbk = MiniBatchKMeans(n_clusters=4, init='k-means++', 
                       batch_size=1024, n_init=10, random_state=42)
mbk.fit(X)

print(f"KMeans SSE:    {kmeans.inertia_:.2f}")
print(f"MiniBatch SSE: {mbk.inertia_:.2f}")

六、如何选择？
表格
场景	推荐
数据量 < 1万，追求最优效果	KMeans
数据量大（数万~百万级），内存/时间受限	MiniBatchKMeans
流式数据 / 在线学习	MiniBatchKMeans（支持 partial_fit）
需要 elkan 加速且内存充足	KMeans（algorithm='elkan'）

一句话总结‌：两者优化目标相同（最小化 SSE），MiniBatchKMeans 用"抽样 + 滑动平均"换取了大幅提速，代价是精度的微小损失——大数据场景下是性价比极高的选择。

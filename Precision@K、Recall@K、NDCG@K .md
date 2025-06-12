在推荐系统中，Precision@K、Recall@K、NDCG@K是用于评估推荐性能的重要指标，以下是它们的含义和具体计算公式：

### Precision@K

* **含义**：Precision@K衡量的是在推荐给用户的前K个物品中，用户实际感兴趣的物品所占的比例。它反映了推荐系统的准确性，即推荐给用户的物品中有多少是用户真正喜欢的。
* **计算公式**：

   \[
   \text{Precision@K} = \frac{\text{在前K个推荐中正确推荐的物品数量}}{K}
   \]

   其中，正确推荐的物品数量指的是用户实际喜欢的物品（相关物品）出现在推荐列表中的数量。

### Recall@K

* **含义**：Recall@K衡量的是在推荐给用户的前K个物品中，覆盖了用户所有感兴趣物品的比例。它反映了推荐系统的召回能力，即推荐系统能够找出用户多少比例的感兴趣物品。
* **计算公式**：

   \[
   \text{Recall@K} = \frac{\text{在前K个推荐中正确推荐的物品数量}}{\text{用户感兴趣的总物品数量}}
   \]

### NDCG@K

* **含义**：NDCG@K（Normalized Discounted Cumulative Gain@K）是一种考虑了推荐物品排序位置的评估指标。它不仅关注推荐物品是否正确，还关注这些物品在推荐列表中的位置。NDCG@K的值越高，说明推荐系统的性能越好。
* **计算公式**：

   NDCG@K的计算相对复杂，它首先计算DCG@K（Discounted Cumulative Gain@K），然后将DCG@K除以IDCG@K（Ideal Discounted Cumulative Gain@K）进行归一化。

   \[
   \text{DCG@K} = \sum_{i=1}^{K} \frac{2^{r_i} - 1}{\log_2(i + 1)}
   \]

   \[
   \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
   \]

   其中，\(r_i\)表示第i个推荐物品的相关性得分（通常根据用户对物品的评分或点击行为来确定），IDCG@K是理想情况下的DCG@K值，即如果推荐列表中的物品完全按照相关性得分从高到低排序，那么此时的DCG@K值就是IDCG@K。

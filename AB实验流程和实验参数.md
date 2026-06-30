## 核心结论

推荐系统实验平台的本质是：**把用户流量稳定地切分到不同实验组，为每组下发不同的推荐策略参数，然后基于用户行为数据评估不同策略的效果，最终决定是否扩流、推全或下线**。

一个典型流程可以拆成：

1. **创建实验空间 / 实验室**
2. **设置实验组**
3. **配置实验参数**
4. **用户请求时命中分桶**
5. **推荐服务读取实验参数并执行不同策略**
6. **打点采集曝光、点击、转化等数据**
7. **计算指标并做实验决策**

---

# 1. 实验平台解决的核心问题

推荐系统经常需要验证：

- 新召回策略是否更好
- 新排序模型是否提升 CTR / CVR
- 多样性策略是否影响点击和转化
- 商品卡片展示时机是否更优
- 某个特征、权重、阈值是否应该上线

但推荐系统直接全量上线风险很高，因为：

- 算法效果无法完全通过离线指标判断
- 用户行为存在随机波动
- 推荐链路涉及召回、粗排、精排、重排、过滤、业务规则等多个环节
- 一个参数变化可能影响点击率、转化率、GMV、用户体验、系统性能等多个指标

因此实验平台通过 **A/B 实验** 将用户随机分为对照组和实验组，对不同用户采用不同算法或配置，再统计两组用户的行为指标进行比较，例如点击率、停留时间、转化率等。[〔1〕](https://blog.csdn.net/qq_41946364/article/details/125920774)[〔4〕](https://www.cnblogs.com/xiaoyunjun/p/11047392.html)

---

# 2. 实验平台的基本原理

## 2.1 用户分流：同一个用户稳定进入同一实验组

实验平台通常会根据用户 ID 做分桶，例如：

```text
bucket = hash(user_id + experiment_id) % 10000
```

然后根据桶号落入不同实验组：

| 桶范围 | 实验组 | 配置 |
|---|---|---|
| 0 - 4999 | 对照组 A | 当前线上策略 |
| 5000 - 9999 | 实验组 B | 新推荐策略 |

这样做的关键是 **稳定性**：

- 同一个用户每次访问都进入同一个实验组
- 避免用户一会儿看到旧策略，一会儿看到新策略
- 保证实验指标可比较

一些实验平台支持多种分桶方式，例如 **UID HASH、UID 分桶、过滤条件分桶**；过滤条件分桶可以基于表达式，例如 `gender=man`，但实际业务中应注意合规和隐私保护。[〔7〕](https://help.aliyun.com/zh/airec/experiment-configuration-and-decision)[〔9〕](https://help.aliyun.com/zh/airec/what-is-pai-rec/user-guide/experiment-configuration-and-decision)

---

## 2.2 实验隔离：避免多个实验互相污染

推荐系统里经常同时跑多个实验，例如：

- 召回实验
- 排序模型实验
- 重排多样性实验
- UI 展示实验
- 运营规则实验

如果多个实验同时影响同一批用户，就可能无法判断指标变化到底来自哪个实验。

因此实验平台通常会设计：

| 隔离方式 | 作用 |
|---|---|
| **实验层** | 不同业务模块分层，如召回层、排序层、UI 层 |
| **实验组** | 同一实验组内多个版本互为对照 |
| **流量桶** | 保证不同实验占用不同流量范围 |
| **人群定向** | 限定实验只对特定用户或场景生效 |
| **白名单 / 调试用户** | 上线前验证逻辑是否正确 |

实验组一般是实验的集合，一个实验组下多个实验版本互为对照，每一份不同配置对应一个实验。[〔7〕](https://help.aliyun.com/zh/airec/experiment-configuration-and-decision)[〔9〕](https://help.aliyun.com/zh/airec/what-is-pai-rec/user-guide/experiment-configuration-and-decision)

---

# 3. 创建实验时，平台内部发生了什么？

以一个“推荐排序模型参数实验”为例。

## 3.1 创建实验室 / 实验空间

你在平台上创建实验时，通常会填写：

- 实验名称
- 业务场景
- 实验负责人
- 实验描述
- 分桶方式
- 是否为 Base 实验室
- 实验生效环境

**Base 实验室**可以理解为兜底配置。当某些流量没有命中任何实验时，会走 Base 配置，避免推荐服务无配置可用。相关实验平台文档中也提到，Base 实验室通常作为兜底实验室，建议至少保留一个。[〔7〕](https://help.aliyun.com/zh/airec/experiment-configuration-and-decision)[〔9〕](https://help.aliyun.com/zh/airec/what-is-pai-rec/user-guide/experiment-configuration-and-decision)

内部原理可以理解为：

```text
实验平台生成一份实验元信息：
experiment_id
scene
owner
bucket_method
status
base_config
```

这些信息会被保存到实验平台的配置中心或数据库中。

---

## 3.2 设置实验组

实验组负责定义：

- 谁参与实验
- 实验流量占比是多少
- 是否需要 AA 实验
- 对照组和实验组分别是什么
- 是否配置调试用户
- 是否配置定向人群

例如：

| 组别 | 流量 | 说明 |
|---|---:|---|
| A 对照组 | 50% | 当前线上模型 |
| B 实验组 | 50% | 新排序模型 |

如果需要先小流量验证，也可以：

| 组别 | 流量 | 说明 |
|---|---:|---|
| A 对照组 | 95% | 当前线上策略 |
| B 实验组 | 5% | 新策略小流量验证 |

平台会为实验组分配桶号。用户请求进入推荐系统时，会根据用户 ID 计算桶号，再判断命中哪个实验组。

---

# 4. 配置实验参数的原理

## 4.1 实验参数是什么？

实验参数本质上是：**平台下发给业务代码的一组配置项**。

例如：

```json
{
  "rank_model": "model_v2",
  "recall_topk": 500,
  "diversity_weight": 0.15,
  "enable_new_feature": true
}
```

推荐服务拿到这些参数后，会执行不同逻辑。

例如：

```python
if exp_config["enable_new_feature"]:
    use_new_feature = True

rank_model = load_model(exp_config["rank_model"])
recall_topk = exp_config["recall_topk"]
diversity_weight = exp_config["diversity_weight"]
```

实验参数通常是对 A/B 实验版本的补充，可以是 Number、String、Boolean、Json 等类型；合理设计实验参数，可以在不频繁改代码的情况下灵活组合不同实验策略。[〔5〕](https://www.modb.pro/db/1753237060924297216)

---

## 4.2 为什么不建议把实验参数设计成“实验组编号”？

不推荐这样设计：

```json
{
  "experiment_mode": 3
}
```

然后代码里写：

```python
if experiment_mode == 0:
    # 对照组
elif experiment_mode == 1:
    # 模型优化
elif experiment_mode == 2:
    # 模型优化 + 引导样式1
elif experiment_mode == 3:
    # 模型优化 + 引导样式2
```

这种方式的问题是：

- 参数语义不清晰
- 复用性差
- 每次新增组合都要改代码
- 后续实验难以扩展
- 容易造成硬编码

更推荐按 **功能控制维度** 设计参数，例如：

```json
{
  "recommend_model_optimize": true,
  "show_interact_guide": true,
  "show_duration": 5,
  "video_play_duration": 10
}
```

这样可以通过参数组合实现多个实验版本，扩展性更好。类似实践中也建议不要按实验组枚举设计参数，而应按功能控制维度设计实验参数。[〔5〕](https://www.modb.pro/db/1753237060924297216)

---

# 5. 推荐请求命中实验的完整链路

一个用户访问推荐页面时，大致链路如下：

```text
用户请求
  ↓
推荐服务接收 user_id、scene、device、page 等上下文
  ↓
请求实验平台 / 本地实验配置缓存
  ↓
根据 user_id + experiment_id 计算分桶
  ↓
判断命中对照组还是实验组
  ↓
返回对应实验参数
  ↓
推荐服务根据参数执行召回、排序、重排等逻辑
  ↓
返回推荐结果
  ↓
客户端曝光 / 点击 / 加购 / 下单等行为打点
  ↓
指标平台统计实验效果
```

可以用伪代码表示：

```python
def recommend(user_id, scene):
    exp_config = experiment_client.get_config(
        user_id=user_id,
        scene=scene,
        experiment_name="rank_model_exp"
    )

    recall_topk = exp_config.get("recall_topk", 300)
    rank_model_name = exp_config.get("rank_model", "model_v1")
    diversity_weight = exp_config.get("diversity_weight", 0.0)

    candidates = recall(user_id, topk=recall_topk)
    ranked_items = rank(user_id, candidates, model=rank_model_name)
    final_items = rerank(ranked_items, diversity_weight=diversity_weight)

    return final_items
```

---

# 6. 实验参数如何影响推荐系统？

推荐系统通常包括：

```text
召回 → 粗排 → 精排 → 重排 → 过滤 → 展示
```

实验参数可以作用在不同阶段。

## 6.1 召回阶段参数

示例：

```json
{
  "recall_topk": 1000,
  "enable_i2i_recall": true,
  "enable_vector_recall": true
}
```

影响：

- 候选商品数量
- 召回通道是否开启
- 召回比例
- 长尾商品覆盖率

---

## 6.2 排序阶段参数

示例：

```json
{
  "rank_model": "rank_model_v2",
  "ctr_weight": 0.6,
  "cvr_weight": 0.3,
  "price_weight": 0.1
}
```

影响：

- 排序模型版本
- 多目标融合权重
- 点击、转化、成交、用户体验之间的平衡

---

## 6.3 重排阶段参数

示例：

```json
{
  "diversity_weight": 0.2,
  "same_brand_limit": 2,
  "same_category_limit": 3
}
```

影响：

- 推荐列表多样性
- 品类打散
- 品牌打散
- 用户探索体验

---

## 6.4 业务规则参数

示例：

```json
{
  "enable_price_filter": true,
  "min_score_threshold": 0.35,
  "blacklist_filter": true
}
```

影响：

- 商品过滤逻辑
- 风控规则
- 业务准入规则
- 低质内容拦截

---

# 7. 指标计算原理

实验上线后，需要看实验组相对对照组是否更好。

常见指标包括：

| 指标类型 | 指标示例 | 含义 |
|---|---|---|
| 用户行为 | CTR、停留时长、互动率 | 用户是否愿意点击和消费 |
| 转化效果 | CVR、加购率、下单率 | 是否促进交易转化 |
| 商业指标 | GMV、收入、ROI | 是否产生业务价值 |
| 推荐质量 | 覆盖率、多样性、新颖性 | 是否改善推荐生态 |
| 系统性能 | TP99、接口耗时、超时率 | 是否影响服务稳定性 |

推荐系统在线实验一般通过点击率、用户停留时间、转化率等指标度量用户满意度或业务效果。[〔4〕](https://www.cnblogs.com/xiaoyunjun/p/11047392.html)

---

# 8. 为什么离线评估不够，还需要实验平台？

离线实验通常是：

```text
历史数据 → 训练集 / 测试集 → 训练模型 → 计算指标
```

优点是：

- 成本低
- 速度快
- 不影响真实用户
- 可以快速筛选算法

但缺点是：

- 无法完全反映真实用户反馈
- 很难评估点击率、转化率等线上商业指标
- 历史数据存在偏差
- 离线指标好，不代表线上一定好

资料中也提到，离线实验不需要真实用户参与，快速方便，但无法计算商业上关心的点击率、转化率等指标，且离线指标和商业指标之间可能存在差距。[〔1〕](https://blog.csdn.net/qq_41946364/article/details/125920774)[〔8〕](https://www.cnblogs.com/dugk/archive/2004/01/13/8900977.html)

所以推荐系统通常采用递进式验证：

```text
离线实验 → 用户调查 / 小流量实验 → 在线 A/B 实验 → 灰度扩流 → 全量发布
```

一般来说，推荐算法更新需要先通过离线实验验证指标，再通过用户调查或小流量验证体验，最后通过在线 A/B 测试确认商业指标是否优于旧算法。[〔4〕](https://www.cnblogs.com/xiaoyunjun/p/11047392.html)[〔8〕](https://www.cnblogs.com/dugk/archive/2004/01/13/8900977.html)

---

# 9. 实验决策原理

实验跑一段时间后，平台会对比：

```text
实验组指标 - 对照组指标
```

例如：

| 指标 | 对照组 | 实验组 | 相对变化 |
|---|---:|---:|---:|
| CTR | 8.00% | 8.32% | +4.00% |
| CVR | 2.50% | 2.55% | +2.00% |
| GMV | 100万 | 103万 | +3.00% |
| TP99 | 120ms | 145ms | 变差 |

决策不只看单一指标，而要综合判断：

- 主指标是否显著正向
- 护栏指标是否变差
- 系统性能是否可接受
- 分人群是否存在明显负向
- 指标是否稳定
- 是否存在节假日、活动、流量波动干扰

如果效果正向且稳定，通常先扩流，再推全；如果效果较差，则下线实验。相关实验流程中也提到，效果正向时可以先调整流量进行扩流，稳定后再推全；效果较差时可以下线。[〔7〕](https://help.aliyun.com/zh/airec/experiment-configuration-and-decision)[〔9〕](https://help.aliyun.com/zh/airec/what-is-pai-rec/user-guide/experiment-configuration-and-decision)

---

# 10. 一个完整例子：推荐多样性实验

## 10.1 实验目标

验证提升推荐多样性是否能提高用户长期体验，同时不明显损害点击和转化。

---

## 10.2 实验设计

| 组别 | 流量 | 参数 |
|---|---:|---|
| 对照组 A | 50% | `diversity_weight=0.0` |
| 实验组 B | 50% | `diversity_weight=0.2` |

参数配置：

```json
{
  "diversity_weight": 0.2,
  "same_category_limit": 3,
  "same_brand_limit": 2
}
```

---

## 10.3 服务执行逻辑

```python
def rerank(items, exp_config):
    diversity_weight = exp_config.get("diversity_weight", 0.0)
    same_category_limit = exp_config.get("same_category_limit", 999)
    same_brand_limit = exp_config.get("same_brand_limit", 999)

    return diversity_rerank(
        items,
        diversity_weight=diversity_weight,
        same_category_limit=same_category_limit,
        same_brand_limit=same_brand_limit
    )
```

---

## 10.4 观察指标

| 指标类型 | 指标 |
|---|---|
| 主指标 | 人均点击、CTR、停留时长 |
| 转化指标 | 加购率、下单率、GMV |
| 推荐质量 | 品类覆盖率、品牌覆盖率、新颖性 |
| 护栏指标 | TP99、接口失败率、投诉率 |

---

## 10.5 决策

| 结果 | 决策 |
|---|---|
| CTR、转化均提升，性能稳定 | 扩流、推全 |
| CTR 略降，但长期留存提升明显 | 结合业务目标综合判断 |
| 点击和转化明显下降 | 下线或调小多样性权重 |
| 性能明显变差 | 优化实现后重新实验 |

---

# 11. 实验平台配置的本质

可以概括为三层：

```text
第一层：流量控制
决定谁进入哪个实验组。

第二层：参数下发
决定这个实验组使用什么推荐配置。

第三层：效果评估
判断该配置是否优于原配置。
```

更工程化地看：

```text
实验平台 = 分桶系统 + 配置系统 + 打点系统 + 指标系统 + 决策系统
```

| 模块 | 作用 |
|---|---|
| 分桶系统 | 保证用户稳定分组 |
| 配置系统 | 给不同实验组下发不同参数 |
| 推荐服务 | 根据参数执行不同推荐逻辑 |
| 打点系统 | 采集曝光、点击、转化等行为 |
| 指标系统 | 计算 CTR、CVR、GMV、延迟等 |
| 决策系统 | 支持扩流、推全、下线 |

---

## 总结

推荐系统实验平台的具体原理是：

1. **用用户 ID 等标识做稳定分桶**
2. **将用户划分到对照组和实验组**
3. **为不同实验组配置不同推荐参数**
4. **推荐服务实时读取参数并执行不同策略**
5. **通过曝光、点击、转化等打点计算实验指标**
6. **根据统计结果决定扩流、推全或下线**

关键点不在于“创建了一个实验”，而在于：**实验平台把算法策略变化配置化、流量切分稳定化、效果评估数据化、上线决策流程化**。

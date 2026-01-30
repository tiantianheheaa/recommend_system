### 自己总结
- **二分类交叉熵 是 多分类交叉熵的特例**，记多分类交叉熵损失就可以。
- 多分类中的np.sum(y_true_onehot * np.log(y_pred_probs) 和 二分类中的 y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred) 是一样的。所以说二分类是多分类的特例。
- 由于多分类的label只有1个类别是1，其他类别都0，所以计算loss只算 label=1的类别的预估值 就可以了。 所以就是**对label=1的类别的模型预估值 取-log就可以了**。
- 正负样本必须有：
    - 如果只有label=1的正样本，就会使得模型预估值在0-1之间越大越好，越接近1越好。导致pctr的均值不在真实ctr均值附近（例如真实ctr是0.5）。
    - 加入label=0的样本，对于这样的样本，模型预估值越偏向0越好。  综合看，如果有label=1和label=0的样本，就会使得模型pctr的平均值和真实ctr的平均值接近。
    - **模型预估的是pctr是在 预估label=1的概率**。  加入label=0的样本，会导致label=1的预估值偏低，从而对label=1的预估值 接近真实ctr。**真实ctr的计算也是label=1的数量/总样本数量**。 **对预估p值的绝对值有要求：接近真实值。这是在做calibration了**。 如果只是做ranking，只比较模型预估p值的相对值排序就可以。
    - 多分类也是这样：softmax后，pctr的范围还是0-1之间。 多分类就不是负样本了，就是label=0,1,2,...,10的样本都需要有（例如一共是10分类）。最终预估的p值在各个label上的平均值 应该是0.1（假设训练样本中label在1-10的样本数量是相等的）。以label=3为例，会有label=3的正样本（10个），也会有label!=3的样本（90个），导致最终在label=3的预估p值的平均值是0.1。

### **二分类交叉熵损失 vs. 多分类交叉熵损失**
交叉熵损失（Cross-Entropy Loss）是分类任务中常用的损失函数，用于衡量模型预测的概率分布与真实标签分布之间的差异。以下是两者的原理、计算过程及代码实现（不调用库）。

---

## **1. 二分类交叉熵损失（Binary Cross-Entropy）**
### **原理**
<img width="1642" height="1246" alt="image" src="https://github.com/user-attachments/assets/f454bb22-1feb-43d8-af86-f189c4b4bc9c" />


### **代码实现**
```python
import numpy as np

def binary_cross_entropy(y_true, y_pred):
    """
    y_true: 真实标签，形状 (N,)，取值 0 或 1
    y_pred: 预测概率，形状 (N,)，取值 [0, 1]
    """
    epsilon = 1e-15  # 避免 log(0) 的数值不稳定问题
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 限制预测值范围
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# 示例
y_true = np.array([1, 0, 1])
y_pred = np.array([0.9, 0.2, 0.8])
print("Binary CE Loss:", binary_cross_entropy(y_true, y_pred))
```

---

## **2. 多分类交叉熵损失（Categorical Cross-Entropy）**
### **原理**

<img width="734" height="603" alt="截屏2026-01-29 21 36 45" src="https://github.com/user-attachments/assets/f8e8684c-5252-4884-bc5a-1671566089d5" />

### **代码实现**
```python
def categorical_cross_entropy(y_true_onehot, y_pred_probs):
    """
    y_true_onehot: 真实标签的 one-hot 编码，形状 (N, C)
    y_pred_probs: 预测概率（Softmax 输出），形状 (N, C)
    """
    epsilon = 1e-15  # 避免 log(0)
    y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon)
    loss = -np.mean(np.sum(y_true_onehot * np.log(y_pred_probs), axis=1))
    return loss

# 示例
y_true_onehot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3 个样本，3 个类别
y_pred_probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
print("Categorical CE Loss:", categorical_cross_entropy(y_true_onehot, y_pred_probs))
```

---

## **3. 关键区别**
| 特性                | 二分类交叉熵                     | 多分类交叉熵                     |
|---------------------|----------------------------------|----------------------------------|
| **适用任务**         | 二分类                           | 多分类                           |
| **输出层激活函数**   | Sigmoid（单个输出）              | Softmax（多个输出）              |
| **标签格式**         | 0 或 1                           | One-hot 编码                     |
| **损失计算**         | 每个样本独立计算                 | 对所有类别求和                   |

---

## **4. 数值稳定性优化**
- **避免 \( \log(0) \)**：用 `np.clip` 限制预测值范围（如 `[1e-15, 1-1e-15]`）。
- **批量计算**：用 `np.mean` 或 `np.sum` 代替循环，提高效率。

---

## **5. 完整示例（从预测到损失计算）**
```python
# 二分类示例（含 Sigmoid）
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

y_true = np.array([1, 0, 1])
logits = np.array([2.0, -1.0, 1.5])  # 模型原始输出（未归一化）
y_pred = sigmoid(logits)  # 转换为概率
print("Binary CE Loss:", binary_cross_entropy(y_true, y_pred))

# 多分类示例（含 Softmax）
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 数值稳定技巧
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

y_true_labels = np.array([0, 1, 2])  # 真实类别索引
y_true_onehot = np.zeros((3, 3))
y_true_onehot[np.arange(3), y_true_labels] = 1  # 转换为 one-hot
logits = np.array([[3.0, 1.0, 0.2], [1.0, 5.0, 0.1], [0.5, 2.0, 1.0]])  # 模型原始输出
y_pred_probs = softmax(logits)  # 转换为概率
print("Categorical CE Loss:", categorical_cross_entropy(y_true_onehot, y_pred_probs))
```

---

### **总结**
- **二分类交叉熵**：适用于两类问题，用 Sigmoid + 单输出。
- **多分类交叉熵**：适用于多类问题，用 Softmax + 多输出。
- **数值稳定**：用 `np.clip` 和 `np.exp(x - max(x))` 避免计算错误。
- **代码实现**：直接操作 NumPy 数组，避免调用深度学习库。

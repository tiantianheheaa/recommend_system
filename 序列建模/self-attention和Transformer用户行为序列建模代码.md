下面给出 **Self-Attention** 与 **Transformer** 两种用户行为序列建模的具体方案、可运行的 PyTorch 代码，并专门讨论召回双塔在线耗时约束下，行为序列长度的合理选择。

---

## 一、Self-Attention 行为序列建模

### 1.1 方案设计
- **输入**：用户最近 N 条行为序列，每条行为表示为物品 Embedding（维度 \(d\)），加上行为类型 Embedding、时间差 Embedding 等。
- **Self-Attention 层**：对序列中每个位置，计算其与所有位置的注意力，捕捉行为间的全局依赖。
- **聚合**：用 Attention Pooling（以可学习 Query 向量或平均池化）将 N 个上下文表示压缩成一个用户兴趣向量。
- **优势**：结构简单，一次 Self-Attention 即可建模行为关联，比 Transformer 更轻量。

### 1.2 代码实现（PyTorch）
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionBehaviorModel(nn.Module):
    def __init__(self, item_emb_dim, behavior_emb_dim=8, pos_emb_dim=16,
                 hidden_dim=128, output_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 行为类型嵌入（点击、点赞、播放时长分桶等）
        self.behavior_type_emb = nn.Embedding(10, behavior_emb_dim)
        # 位置嵌入（时间差用正弦编码或可学习嵌入，这里用可学习）
        self.pos_emb = nn.Embedding(500, pos_emb_dim)  # 假设最大序列长度500
        # 输入变换到 hidden_dim
        self.input_proj = nn.Linear(item_emb_dim + behavior_emb_dim + pos_emb_dim, hidden_dim)

        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                               dropout=dropout, batch_first=True)
        # 注意力后的小型前馈网络 (可选)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 聚合层：用可学习 Query 做注意力池化
        self.attn_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn_pool = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1,
                                               batch_first=True)
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, item_emb_seq, behavior_type_seq, position_seq, mask=None):
        """
        item_emb_seq: [B, N, d]  行为物品的embedding
        behavior_type_seq: [B, N]  行为类型索引
        position_seq: [B, N]      位置/时间差索引
        mask: [B, N]  bool，True表示需要mask的位置 (padding)
        """
        B, N, _ = item_emb_seq.shape

        # 行为类型嵌入
        beh_emb = self.behavior_type_emb(behavior_type_seq)  # [B, N, beh_dim]
        # 位置嵌入
        pos_emb = self.pos_emb(position_seq)  # [B, N, pos_dim]
        # 拼接并变换
        x = torch.cat([item_emb_seq, beh_emb, pos_emb], dim=-1)  # [B, N, in_dim]
        x = self.input_proj(x)  # [B, N, hidden_dim]

        # 自注意力
        attn_mask = None
        if mask is not None:
            # 将 True(mask) 转换为 -inf 用于 attention
            attn_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
            # MultiheadAttention 需要 [N, N] 或 [B*num_heads, N, N]，但 batch_first 下支持 [B, N, N]？ 
            # 通常传入 bool mask，这里简单处理：将 mask 扩展为 [B, N, N] 的key padding mask
            # 在batch_first=True下，key_padding_mask 可以是 [B, N]
            key_padding_mask = mask  # True表示pad
        else:
            key_padding_mask = None

        # 自注意力
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)  # [B, N, hidden_dim]
        x = self.layer_norm1(x + self.dropout(attn_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_out))  # [B, N, hidden_dim]

        # 聚合：使用可学习 query 对所有位置做注意力
        query = self.attn_query.expand(B, -1, -1)  # [B, 1, hidden_dim]
        # 注意 mask: 如果某位置需要忽略，需将对应 score 置为 -inf
        if key_padding_mask is not None:
            # 转换为 [B, 1, N] 的 attn_mask
            attn_mask_pool = key_padding_mask.unsqueeze(1)  # [B, 1, N] True -> mask
            # MultiheadAttention 的 attn_mask 逻辑较复杂，简化：手动加负无穷
            # 这里用 torch.baddbmm 等手动实现注意力，或利用 MHA 但需要设置 key_padding_mask
            # 使用 MHA 时，直接传 key_padding_mask 即可
            user_vec, _ = self.attn_pool(query, x, x, key_padding_mask=key_padding_mask)
        else:
            user_vec, _ = self.attn_pool(query, x, x)  # [B, 1, hidden_dim]

        user_vec = user_vec.squeeze(1)  # [B, hidden_dim]
        user_vec = self.output_layer(user_vec)  # [B, output_dim]
        return user_vec
```

**使用示例**：
```python
model = SelfAttentionBehaviorModel(item_emb_dim=128)
# 模拟输入：batch=2, seq_len=50, item_emb_dim=128
item_seq = torch.randn(2, 50, 128)
beh_type = torch.randint(0, 10, (2, 50))
pos = torch.arange(50).unsqueeze(0).expand(2, -1)
mask = torch.zeros(2, 50, dtype=torch.bool)  # 无padding
user_vector = model(item_seq, beh_type, pos, mask)  # [2, 128]
```

---

## 二、Transformer 行为序列建模

### 2.1 方案设计
- 与 Self-Attention 方案类似，但使用**多层 Transformer Encoder**，每层含多头自注意力 + 前馈网络，通过残差连接和 LayerNorm 增强表达能力。
- 通常在输入前加**位置编码**（可学习或正弦），因为自注意力本身无序。
- 聚合同样使用 Attention Pooling，但可以利用 Transformer 输出的最后一个位置的向量（如果序列填充在右侧）或平均池化。
- 相比单层 Self-Attention，Transformer 能提取更深层次的行为模式，但计算量和参数量增大。

### 2.2 代码实现
```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x):
        # x: [B, N, d_model]
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class TransformerBehaviorModel(nn.Module):
    def __init__(self, item_emb_dim, behavior_emb_dim=8, hidden_dim=128,
                 output_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.behavior_type_emb = nn.Embedding(10, behavior_emb_dim)
        # 输入投影
        self.input_proj = nn.Linear(item_emb_dim + behavior_emb_dim, hidden_dim)
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=500)
        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*4,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 聚合层：Attention Pooling
        self.attn_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn_pool = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, item_emb_seq, behavior_type_seq, mask=None):
        """
        item_emb_seq: [B, N, item_dim]
        behavior_type_seq: [B, N]
        mask: [B, N] True 表示 padding
        """
        B, N, _ = item_emb_seq.shape
        beh_emb = self.behavior_type_emb(behavior_type_seq)  # [B, N, beh_dim]
        x = torch.cat([item_emb_seq, beh_emb], dim=-1)
        x = self.input_proj(x)  # [B, N, hidden_dim]
        x = self.pos_encoder(x)

        # Transformer 需要 key_padding_mask (True -> 忽略)
        # 如果mask为True表示padding，则直接传入
        x = self.transformer(x, src_key_padding_mask=mask)  # [B, N, hidden_dim]

        # 聚合
        query = self.attn_query.expand(B, -1, -1)
        if mask is not None:
            user_vec, _ = self.attn_pool(query, x, x, key_padding_mask=mask)
        else:
            user_vec, _ = self.attn_pool(query, x, x)
        user_vec = user_vec.squeeze(1)  # [B, hidden_dim]
        user_vec = self.output_layer(user_vec)
        return user_vec
```

---

## 三、召回双塔耗时约束下，序列长度如何选择？

### 3.1 关键耗时分析
在召回双塔中，用户塔通常需要**在线实时计算**（因为用户最新行为实时更新），而物品塔可以离线预计算。用户塔的推理延迟直接影响整体召回 P99 延迟，必须严格控制在 5~20ms 内（视系统要求）。

Self-Attention / Transformer 的复杂度主要取决于序列长度 \(N\)：
- **Self-Attention**：计算量 \(O(N^2 \cdot d)\)，内存 \(O(N^2)\)，当 \(N>500\) 时 GPU 推理延迟可能超过 10ms。
- **Transformer**：多层 \(O(L \cdot N^2 \cdot d)\)，更长时更重。
- 线上服务通常用 **CPU 或轻量 GPU（如 T4）** 推理，对序列长度更为敏感。

### 3.2 合理序列长度建议

| 方案 | 推荐长度 | 理由 |
|------|----------|------|
| **Self-Attention (单层)** | **50~200** | 单层下计算开销可控，200 长度在多数服务器上推理延迟 < 5ms（CPU 优化后）。若行为丰富，可截断近期 200 条；若序列极长，可先用时间衰减过滤到 200 再建模 |
| **Transformer (2层)** | **50~100** | 多层显著增加计算量，100 长度已能覆盖近期关键行为，且延迟可接受。不建议超过 150，否则用户塔可能成为瓶颈 |
| **Transformer (≥3层)** | **≤ 50** | 仅用于极致性能要求不高的离线预计算场景。若需在线，必须配合 **SIM 硬检索** 或 **压缩记忆** 等方法将长序列压缩至短序列 |

### 3.3 工业落地技巧
- **长期序列压缩**：对于几千条长期行为，用时间衰减或行为强度筛选 Top-K（如 200）送入模型，既保留多样兴趣又控制长度。
- **分塔分离**：如果用户塔必须处理超长序列（如数千），可考虑将用户塔拆成“离线长期编码器+在线短期编码器”，在线仅处理最近 50 条，长期兴趣向量离线每日更新，在线拼接。
- **序列填充统一**：将变长序列 padding 到固定最大长度，用 mask 忽略填充位，能利用 GPU 并行，减少动态分支。
- **推理优化**：导出模型到 ONNX / TensorRT，用 FP16 或 INT8 量化，可大幅降低 Self-Attention 矩阵乘法延迟，使得长度 200 的 Transformer 也能在 10ms 内完成。

### 3.4 具体方案组合
**推荐方案**：  
- **在线用户塔**：采用 **单层 Self-Attention + 行为类型加权**，序列长度固定 100（近 100 次播放行为）。  
- **长期兴趣补充**：离线每天用 Transformer 对全量长期行为（数千）训练并生成多兴趣向量，存入向量引擎，在线与短期兴趣向量合并检索。  

这样既保证了实时性，又充分利用了长历史信息，是多数短视频召回系统的平衡实践。

---

## 四、总结
- Self-Attention 单层方案适合作为在线用户塔，轻量且能捕捉行为关联，序列长度 100~200 性价比最高。
- Transformer 多层方案更适合离线建模或作为复杂兴趣提取器，在线使用时需严格控制长度在 50~100 并配合优化。
- 核心是 **长短序列分离**：在线用短序列实时响应，离线用长序列全面挖掘，两者融合表达用户完整兴趣。

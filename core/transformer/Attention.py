# @Author  : Edlison
# @Date    : 9/24/20 19:43

import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product attention mechanism.
    """

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        forward.

        Args:
            q: Query Tensor, Shape [batch_size, L_q, D_q]
            k: Key Tensor, Shape [batch_size, L_k, D_k]
            v: Value Tensor, Shape [batch_size, L_v, D_v]
            scale: 缩放因子 浮点标量
            attn_mask: Masking Tensor, Shape [batch_size, L_q, L_k]

        Returns:
            上下文Tensor, Attention Tensor.

        @Author  : Edlison
        @Date    : 9/24/20 19:52
        """

        attention = torch.bmm(q, k.transpose(1, 2))  # k 的 1, 2维进行转换
        # attention Shape [batch_size, 1, 1] ?

        if scale:
            # 缩放 论文中除以 根号dk
            attention = attention * scale

        if attn_mask:
            # 需要mask的地方 这只一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)

        # 计算 softmax
        attention = self.softmax(attention)
        # 添加 dropout
        attention = self.dropout(attention)
        # 和 v 做点积
        context = torch.bmm(attention, v)  # [1, 1] * [1, 64]
        # context Shape [batch_size, 1, 64] ?

        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads  # 取整
        self.num_heads = num_heads
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        # multi-head attention 之后要做 layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, q, k, v, attn_mask=None):
        # 残差连接
        residual = q
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = k.size(0)

        # linear投影
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # split by heads
        q = q.view(batch_size * num_heads, -1, dim_per_head)
        k = k.view(batch_size * num_heads, -1, dim_per_head)
        v = v.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # scaled dot-product attention
        scale = (q.size(-1)) ** -0.5
        context, attention = self.dot_product_attention(
            q, k, v, scale, attn_mask
        )

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention

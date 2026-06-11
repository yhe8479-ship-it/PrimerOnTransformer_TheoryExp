import copy
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class TokenEmbedding(nn.Module):
    """
    token id -> embedding vector。
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码。
    输入输出形状: [batch, seq_len, d_model]
    """
    def __init__(self, d_model: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
):
    """
    Scaled Dot-Product Attention。
    query/key/value: [batch, heads, seq_len, d_k]
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    多头注意力。
    """
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        query, key, value = [
            layer(x)
            .view(batch_size, -1, self.n_heads, self.d_k)
            .transpose(1, 2)
            for layer, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_k)
        )

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    逐位置前馈网络。
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    残差连接 + LayerNorm。
    使用 Pre-LN：
        x + Dropout(Sublayer(LayerNorm(x)))
    """
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    一个 Encoder 层：
        self-attention -> feed-forward
    """
    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.sublayers[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """
    N 层 Encoder。
    """
    def __init__(self, layer: EncoderLayer, n_layers: int, d_model: int):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    一个 Decoder 层。

    use_cross_attention=True:
        masked self-attention -> cross-attention -> feed-forward

    use_cross_attention=False:
        masked self-attention -> feed-forward
    """
    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadedAttention,
        src_attn: Optional[MultiHeadedAttention],
        feed_forward: PositionwiseFeedForward,
        dropout: float,
        use_cross_attention: bool,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.use_cross_attention = use_cross_attention

        n_sublayers = 3 if use_cross_attention else 2
        self.sublayers = clones(SublayerConnection(d_model, dropout), n_sublayers)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor],
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        if self.use_cross_attention:
            if memory is None:
                raise ValueError("use_cross_attention=True 时必须传入 memory")
            x = self.sublayers[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
            x = self.sublayers[2](x, self.feed_forward)
        else:
            x = self.sublayers[1](x, self.feed_forward)

        return x


class Decoder(nn.Module):
    """
    N 层 Decoder。
    """
    def __init__(self, layer: DecoderLayer, n_layers: int, d_model: int):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor],
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

from typing import Optional

import torch
from torch import nn

from .config import FlexibleTransformerConfig
from .blocks import (
    TokenEmbedding,
    PositionalEncoding,
    MultiHeadedAttention,
    PositionwiseFeedForward,
    EncoderLayer,
    Encoder,
    DecoderLayer,
    Decoder,
)


class Generator(nn.Module):
    """
    生成头：把 hidden states 映射到词表 logits。
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FlexibleTransformer(nn.Module):
    """
    开关式 Transformer 主干。
    """
    def __init__(self, config: FlexibleTransformerConfig):
        super().__init__()
        self.config = config

        if not config.use_encoder and not config.use_decoder:
            raise ValueError("use_encoder 和 use_decoder 至少要打开一个")

        self.src_embed = None
        self.encoder = None
        if config.use_encoder:
            self.src_embed = nn.Sequential(
                TokenEmbedding(config.src_vocab_size, config.d_model),
                PositionalEncoding(config.d_model, config.dropout, config.max_len),
            )
            enc_attn = MultiHeadedAttention(config.n_heads, config.d_model, config.dropout)
            enc_ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
            enc_layer = EncoderLayer(config.d_model, enc_attn, enc_ff, config.dropout)
            self.encoder = Encoder(enc_layer, config.n_layers, config.d_model)

        self.tgt_embed = None
        self.decoder = None
        if config.use_decoder:
            self.tgt_embed = nn.Sequential(
                TokenEmbedding(config.tgt_vocab_size, config.d_model),
                PositionalEncoding(config.d_model, config.dropout, config.max_len),
            )

            use_cross_attention = config.use_encoder and config.use_decoder

            dec_self_attn = MultiHeadedAttention(config.n_heads, config.d_model, config.dropout)
            dec_src_attn = (
                MultiHeadedAttention(config.n_heads, config.d_model, config.dropout)
                if use_cross_attention
                else None
            )
            dec_ff = PositionwiseFeedForward(config.d_model, config.d_ff, config.dropout)
            dec_layer = DecoderLayer(
                d_model=config.d_model,
                self_attn=dec_self_attn,
                src_attn=dec_src_attn,
                feed_forward=dec_ff,
                dropout=config.dropout,
                use_cross_attention=use_cross_attention,
            )
            self.decoder = Decoder(dec_layer, config.n_layers, config.d_model)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("当前模型没有打开 Encoder")
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        memory: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("当前模型没有打开 Decoder")
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(
        self,
        src: Optional[torch.Tensor] = None,
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.use_encoder and self.config.use_decoder:
            if src is None or tgt is None:
                raise ValueError("Encoder-Decoder 模式必须同时传入 src 和 tgt")
            memory = self.encode(src, src_mask)
            return self.decode(tgt, tgt_mask, memory=memory, src_mask=src_mask)

        if self.config.use_encoder:
            if src is None:
                raise ValueError("Encoder-only 模式必须传入 src")
            return self.encode(src, src_mask)

        if self.config.use_decoder:
            if tgt is None:
                raise ValueError("Decoder-only 模式必须传入 tgt")
            return self.decode(tgt, tgt_mask, memory=None, src_mask=None)

        raise RuntimeError("不应该运行到这里")

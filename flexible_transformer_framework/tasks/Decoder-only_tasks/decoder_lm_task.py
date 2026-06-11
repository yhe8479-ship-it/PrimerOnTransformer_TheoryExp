import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
import torch.nn.functional as F

from model import FlexibleTransformer, FlexibleTransformerConfig, Generator
from model.masks import make_tgt_mask


class DecoderOnlyLanguageModel(nn.Module):
    """
    Decoder-only 任务 1：语言模型 / 文本续写。

    训练方式：
        输入：前面的 token
        输出：下一个 token
    """
    def __init__(self, vocab_size: int, d_model: int = 256, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        config = FlexibleTransformerConfig(
            use_encoder=False,
            use_decoder=True,
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=d_model,
            n_layers=4,
            n_heads=8,
            d_ff=1024,
            dropout=0.1,
            pad_id=pad_id,
        )
        self.backbone = FlexibleTransformer(config)
        self.generator = Generator(d_model, vocab_size)

    def forward(self, tgt_in: torch.Tensor) -> torch.Tensor:
        tgt_mask = make_tgt_mask(tgt_in, self.pad_id)
        hidden = self.backbone(tgt=tgt_in, tgt_mask=tgt_mask)
        return self.generator(hidden)


def demo():
    torch.manual_seed(0)
    batch_size, seq_len = 4, 12
    vocab_size = 1000
    pad_id = 0

    model = DecoderOnlyLanguageModel(vocab_size, pad_id=pad_id)
    tgt_in = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = torch.randint(1, vocab_size, (batch_size, seq_len))

    logits = model(tgt_in)
    loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=pad_id)
    print("[Decoder-only/LM] logits:", tuple(logits.shape), "loss:", float(loss))


if __name__ == "__main__":
    demo()

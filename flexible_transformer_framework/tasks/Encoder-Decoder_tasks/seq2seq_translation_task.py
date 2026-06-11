import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
import torch.nn.functional as F

from model import FlexibleTransformer, FlexibleTransformerConfig, Generator
from model.masks import make_src_mask, make_tgt_mask


class Seq2SeqTranslationModel(nn.Module):
    """
    Encoder-Decoder 任务 1：机器翻译。

    例子：
        src: 中文
        tgt: 英文
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        config = FlexibleTransformerConfig(
            use_encoder=True,
            use_decoder=True,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_layers=4,
            n_heads=8,
            d_ff=1024,
            dropout=0.1,
            pad_id=pad_id,
        )
        self.backbone = FlexibleTransformer(config)
        self.generator = Generator(d_model, tgt_vocab_size)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_mask = make_src_mask(src, self.pad_id)
        tgt_mask = make_tgt_mask(tgt_in, self.pad_id)
        hidden = self.backbone(src=src, tgt=tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.generator(hidden)


def demo():
    torch.manual_seed(0)
    batch_size, src_len, tgt_len = 4, 10, 12
    src_vocab_size, tgt_vocab_size = 1200, 1000
    pad_id = 0

    model = Seq2SeqTranslationModel(src_vocab_size, tgt_vocab_size, pad_id=pad_id)
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt_in = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    labels = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))

    logits = model(src, tgt_in)
    loss = F.cross_entropy(logits.view(-1, tgt_vocab_size), labels.view(-1), ignore_index=pad_id)
    print("[Encoder-Decoder/Translation] logits:", tuple(logits.shape), "loss:", float(loss))


if __name__ == "__main__":
    demo()

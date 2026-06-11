import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
import torch.nn.functional as F

from model import FlexibleTransformer, FlexibleTransformerConfig, Generator
from model.masks import make_src_mask, make_tgt_mask


class SummarizationModel(nn.Module):
    """
    Encoder-Decoder 任务 2：文本摘要。

    例子：
        src: 长文章
        tgt: 短摘要

    和翻译结构几乎一样，区别主要是训练数据不同。
    """
    def __init__(
        self,
        src_vocab_size: int,
        summary_vocab_size: int,
        d_model: int = 256,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        config = FlexibleTransformerConfig(
            use_encoder=True,
            use_decoder=True,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=summary_vocab_size,
            d_model=d_model,
            n_layers=4,
            n_heads=8,
            d_ff=1024,
            dropout=0.1,
            pad_id=pad_id,
        )
        self.backbone = FlexibleTransformer(config)
        self.generator = Generator(d_model, summary_vocab_size)

    def forward(self, article_tokens: torch.Tensor, summary_in: torch.Tensor) -> torch.Tensor:
        src_mask = make_src_mask(article_tokens, self.pad_id)
        tgt_mask = make_tgt_mask(summary_in, self.pad_id)
        hidden = self.backbone(
            src=article_tokens,
            tgt=summary_in,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )
        return self.generator(hidden)


def demo():
    torch.manual_seed(0)
    batch_size, article_len, summary_len = 4, 20, 8
    src_vocab_size, summary_vocab_size = 1500, 1000
    pad_id = 0

    model = SummarizationModel(src_vocab_size, summary_vocab_size, pad_id=pad_id)
    article_tokens = torch.randint(1, src_vocab_size, (batch_size, article_len))
    summary_in = torch.randint(1, summary_vocab_size, (batch_size, summary_len))
    labels = torch.randint(1, summary_vocab_size, (batch_size, summary_len))

    logits = model(article_tokens, summary_in)
    loss = F.cross_entropy(logits.view(-1, summary_vocab_size), labels.view(-1), ignore_index=pad_id)
    print("[Encoder-Decoder/Summarization] logits:", tuple(logits.shape), "loss:", float(loss))


if __name__ == "__main__":
    demo()

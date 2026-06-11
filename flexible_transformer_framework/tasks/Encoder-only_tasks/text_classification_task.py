import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
import torch.nn.functional as F

from model import FlexibleTransformer, FlexibleTransformerConfig
from model.masks import make_src_mask


class TextClassificationModel(nn.Module):
    """
    Encoder-only 任务 1：文本分类 / 情感分析。

    例子：
        输入：这个电影太好看了
        输出：正面
    """
    def __init__(self, vocab_size: int, num_classes: int, d_model: int = 256, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        config = FlexibleTransformerConfig(
            use_encoder=True,
            use_decoder=False,
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
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src_mask = make_src_mask(src, self.pad_id)
        hidden = self.backbone(src=src, src_mask=src_mask)  # [B, S, D]

        valid = (src != self.pad_id).unsqueeze(-1).float()
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        return self.classifier(pooled)


def demo():
    torch.manual_seed(0)
    batch_size, seq_len = 4, 12
    vocab_size, num_classes = 1000, 3

    model = TextClassificationModel(vocab_size, num_classes)
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, num_classes, (batch_size,))

    logits = model(src)
    loss = F.cross_entropy(logits, labels)
    print("[Encoder-only/TextClassification] logits:", tuple(logits.shape), "loss:", float(loss))


if __name__ == "__main__":
    demo()

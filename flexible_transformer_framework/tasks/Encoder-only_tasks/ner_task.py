import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
import torch.nn.functional as F

from model import FlexibleTransformer, FlexibleTransformerConfig
from model.masks import make_src_mask


class NERModel(nn.Module):
    """
    Encoder-only 任务 2：命名实体识别 / 序列标注。

    例子：
        输入：我 明天 去 北京
        输出：O TIME O LOC
    """
    def __init__(self, vocab_size: int, num_labels: int, d_model: int = 256, pad_id: int = 0):
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
        self.token_classifier = nn.Linear(d_model, num_labels)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src_mask = make_src_mask(src, self.pad_id)
        hidden = self.backbone(src=src, src_mask=src_mask)
        return self.token_classifier(hidden)


def demo():
    torch.manual_seed(0)
    batch_size, seq_len = 4, 12
    vocab_size, num_labels = 1000, 7
    pad_id = 0

    model = NERModel(vocab_size, num_labels, pad_id=pad_id)
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, num_labels, (batch_size, seq_len))

    logits = model(src)
    loss = F.cross_entropy(logits.view(-1, num_labels), labels.view(-1), ignore_index=pad_id)
    print("[Encoder-only/NER] logits:", tuple(logits.shape), "loss:", float(loss))


if __name__ == "__main__":
    demo()

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
import torch.nn.functional as F

from model import FlexibleTransformer, FlexibleTransformerConfig, Generator
from model.masks import make_tgt_mask


class DialogueGenerationModel(nn.Module):
    """
    Decoder-only 任务 2：对话生成。

    训练时可以把多轮对话拼成一个序列：
        用户：你好 <SEP> 助手：你好呀 <SEP> 用户：今天开心吗 <SEP> 助手：
    然后让模型继续预测后面的回复 token。

    本质仍然是 GPT 风格的 next-token prediction。
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

    def forward(self, dialogue_tokens: torch.Tensor) -> torch.Tensor:
        tgt_mask = make_tgt_mask(dialogue_tokens, self.pad_id)
        hidden = self.backbone(tgt=dialogue_tokens, tgt_mask=tgt_mask)
        return self.generator(hidden)


def demo():
    torch.manual_seed(0)
    batch_size, seq_len = 4, 16
    vocab_size = 1000
    pad_id = 0

    model = DialogueGenerationModel(vocab_size, pad_id=pad_id)

    # 假设这里已经是 “用户/助手/分隔符” 拼接后的 token 序列。
    dialogue_tokens = torch.randint(1, vocab_size, (batch_size, seq_len))
    labels = torch.randint(1, vocab_size, (batch_size, seq_len))

    logits = model(dialogue_tokens)
    loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=pad_id)
    print("[Decoder-only/DialogueGeneration] logits:", tuple(logits.shape), "loss:", float(loss))


if __name__ == "__main__":
    demo()

from dataclasses import dataclass


@dataclass
class FlexibleTransformerConfig:
    """
    开关式 Transformer 配置。

    use_encoder=True, use_decoder=False:
        Encoder-only，例如文本分类、NER。

    use_encoder=False, use_decoder=True:
        Decoder-only，例如语言模型、文本续写、对话生成。

    use_encoder=True, use_decoder=True:
        Encoder-Decoder，例如翻译、摘要、改写。
    """
    use_encoder: bool = True
    use_decoder: bool = True

    src_vocab_size: int = 32000
    tgt_vocab_size: int = 32000

    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    n_layers: int = 4
    dropout: float = 0.1
    max_len: int = 512

    pad_id: int = 0

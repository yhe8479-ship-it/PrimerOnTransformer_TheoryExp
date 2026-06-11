import torch


def make_src_mask(src: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """
    Encoder 输入 mask。
    src: [batch, src_len]
    return: [batch, 1, 1, src_len]
    True 表示可见，False 表示 padding 位置不可见。
    """
    return (src != pad_id).unsqueeze(1).unsqueeze(2)


def subsequent_mask(size: int, device=None) -> torch.Tensor:
    """
    因果 mask，保证 Decoder 当前位置不能看未来位置。
    return: [1, 1, size, size]
    """
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def make_tgt_mask(tgt: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """
    Decoder 输入 mask = padding mask + causal mask。
    tgt: [batch, tgt_len]
    return: [batch, 1, tgt_len, tgt_len]
    """
    _, tgt_len = tgt.size()
    pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
    causal_mask = subsequent_mask(tgt_len, device=tgt.device)  # [1, 1, T, T]
    return pad_mask & causal_mask

from pathlib import Path
import torch


def require_checkpoint(path, train_hint):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"没有找到权重文件：{path}\n"
            f"请先训练：{train_hint}"
        )
    return path


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def greedy_decode_seq2seq(model, src_ids, src_tokenizer, tgt_tokenizer, max_len=80, device="cpu"):
    """
    Encoder-Decoder 贪心解码：
        1. Encoder 先编码 src
        2. Decoder 从 <BOS> 开始
        3. 每次取最后一个位置 logits 最大的 token
        4. 生成 <EOS> 或达到 max_len 后停止
    """
    from model.masks import make_src_mask, make_tgt_mask

    model.eval()
    model.to(device)

    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = make_src_mask(src, src_tokenizer.pad_id)

    with torch.no_grad():
        memory = model.backbone.encode(src, src_mask)

        ys = torch.tensor([[tgt_tokenizer.bos_id]], dtype=torch.long, device=device)

        for _ in range(max_len):
            tgt_mask = make_tgt_mask(ys, tgt_tokenizer.pad_id)
            hidden = model.backbone.decode(
                tgt=ys,
                tgt_mask=tgt_mask,
                memory=memory,
                src_mask=src_mask,
            )
            logits = model.generator(hidden[:, -1, :])
            next_id = int(logits.argmax(dim=-1).item())

            ys = torch.cat(
                [ys, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )

            if next_id == tgt_tokenizer.eos_id:
                break

    return ys[0].tolist()


def greedy_generate_decoder_only(model, input_ids, tokenizer, max_new_tokens=50, device="cpu"):
    """
    Decoder-only 贪心生成：
        给定前文 input_ids，逐个预测下一个 token。
    """
    model.eval()
    model.to(device)

    ys = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ys)
            next_id = int(logits[:, -1, :].argmax(dim=-1).item())

            ys = torch.cat(
                [ys, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )

            if next_id == tokenizer.eos_id:
                break

    return ys[0].tolist()

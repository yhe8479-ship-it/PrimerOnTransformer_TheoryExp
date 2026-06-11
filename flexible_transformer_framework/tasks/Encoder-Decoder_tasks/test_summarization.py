import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarization_task import SummarizationModel
from utils.simple_tokenizer import CharTokenizer
from utils.inference_utils import require_checkpoint, load_checkpoint, greedy_decode_seq2seq


def summarize(text, max_len):
    run_dir = PROJECT_ROOT / "run" / "summarization"
    ckpt_path = require_checkpoint(
        run_dir / "model.pt",
        "python tasks/Encoder-Decoder_tasks/train_summarization.py --epochs 40",
    )
    src_tok_path = run_dir / "src_tokenizer.json"
    tgt_tok_path = run_dir / "tgt_tokenizer.json"

    src_tok = CharTokenizer.load(src_tok_path)
    tgt_tok = CharTokenizer.load(tgt_tok_path)
    ckpt = load_checkpoint(ckpt_path)

    model = SummarizationModel(
        src_vocab_size=len(src_tok),
        summary_vocab_size=len(tgt_tok),
        d_model=64,
        pad_id=src_tok.pad_id,
    )
    model.load_state_dict(ckpt["model_state"])

    src_ids = src_tok.encode(text, add_eos=True)
    output_ids = greedy_decode_seq2seq(
        model,
        src_ids,
        src_tok,
        tgt_tok,
        max_len=max_len,
    )

    print("原文输入:", text)
    print("摘要输出:", tgt_tok.decode(output_ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default="今天北京天气晴朗，很多市民来到公园散步和运动，城市交通整体平稳。",
    )
    parser.add_argument("--max_len", type=int, default=50)
    args = parser.parse_args()
    summarize(args.text, args.max_len)


if __name__ == "__main__":
    main()

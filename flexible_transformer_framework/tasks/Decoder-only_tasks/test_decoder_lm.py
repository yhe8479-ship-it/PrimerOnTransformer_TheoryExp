import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decoder_lm_task import DecoderOnlyLanguageModel
from utils.simple_tokenizer import CharTokenizer
from utils.inference_utils import require_checkpoint, load_checkpoint, greedy_generate_decoder_only


def generate(prompt, max_new_tokens):
    run_dir = PROJECT_ROOT / "run" / "decoder_lm"
    ckpt_path = require_checkpoint(
        run_dir / "model.pt",
        "python tasks/Decoder-only_tasks/train_decoder_lm.py --epochs 30",
    )
    tok_path = run_dir / "tokenizer.json"

    tokenizer = CharTokenizer.load(tok_path)
    ckpt = load_checkpoint(ckpt_path)

    model = DecoderOnlyLanguageModel(
        vocab_size=len(tokenizer),
        d_model=64,
        pad_id=tokenizer.pad_id,
    )
    model.load_state_dict(ckpt["model_state"])

    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    output_ids = greedy_generate_decoder_only(
        model,
        input_ids,
        tokenizer,
        max_new_tokens=max_new_tokens,
    )

    print("输入前文:", prompt)
    print("生成结果:", tokenizer.decode(output_ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="我明天")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    args = parser.parse_args()
    generate(args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    main()

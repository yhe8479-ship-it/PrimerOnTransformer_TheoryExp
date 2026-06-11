import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dialogue_generation_task import DialogueGenerationModel
from utils.simple_tokenizer import CharTokenizer
from utils.inference_utils import require_checkpoint, load_checkpoint, greedy_generate_decoder_only


def chat(context, max_new_tokens):
    run_dir = PROJECT_ROOT / "run" / "dialogue_generation"
    ckpt_path = require_checkpoint(
        run_dir / "model.pt",
        "python tasks/Decoder-only_tasks/train_dialogue_generation.py --epochs 30",
    )
    tok_path = run_dir / "tokenizer.json"

    tokenizer = CharTokenizer.load(tok_path)
    ckpt = load_checkpoint(ckpt_path)

    model = DialogueGenerationModel(
        vocab_size=len(tokenizer),
        d_model=64,
        pad_id=tokenizer.pad_id,
    )
    model.load_state_dict(ckpt["model_state"])

    input_ids = tokenizer.encode(context, add_bos=True, add_eos=False) + [tokenizer.sep_id]
    output_ids = greedy_generate_decoder_only(
        model,
        input_ids,
        tokenizer,
        max_new_tokens=max_new_tokens,
    )

    generated_reply_ids = output_ids[len(input_ids):]
    print("用户输入:", context)
    print("模型回复:", tokenizer.decode(generated_reply_ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=str, default="你好")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    args = parser.parse_args()
    chat(args.context, args.max_new_tokens)


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from ner_task import NERModel
from utils.simple_tokenizer import CharTokenizer
from utils.training_utils import pad_1d
from utils.inference_utils import require_checkpoint, load_checkpoint


def predict(text):
    run_dir = PROJECT_ROOT / "run" / "ner"
    ckpt_path = require_checkpoint(
        run_dir / "model.pt",
        "python tasks/Encoder-only_tasks/train_ner.py --epochs 30",
    )
    tok_path = run_dir / "tokenizer.json"

    tokenizer = CharTokenizer.load(tok_path)
    ckpt = load_checkpoint(ckpt_path)
    label_to_id = ckpt["label_to_id"]
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    model = NERModel(
        vocab_size=len(tokenizer),
        num_labels=len(label_to_id),
        d_model=64,
        pad_id=tokenizer.pad_id,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = pad_1d([[tokenizer.token_to_id.get(ch, tokenizer.unk_id) for ch in text]], tokenizer.pad_id)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=-1)[0].tolist()

    print("输入文本:", text)
    print("预测标签:")
    for ch, label_id in zip(text, pred[: len(text)]):
        print(f"  {ch}\t{id_to_label[int(label_id)]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="小明明天去北京")
    args = parser.parse_args()
    predict(args.text)


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from text_classification_task import TextClassificationModel
from utils.simple_tokenizer import CharTokenizer
from utils.training_utils import pad_1d
from utils.inference_utils import require_checkpoint, load_checkpoint


def predict(text):
    run_dir = PROJECT_ROOT / "run" / "text_classification"
    ckpt_path = require_checkpoint(
        run_dir / "model.pt",
        "python tasks/Encoder-only_tasks/train_text_classification.py --epochs 30",
    )
    tok_path = run_dir / "tokenizer.json"

    tokenizer = CharTokenizer.load(tok_path)
    ckpt = load_checkpoint(ckpt_path)
    label_to_id = ckpt["label_to_id"]
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    model = TextClassificationModel(
        vocab_size=len(tokenizer),
        num_classes=len(label_to_id),
        d_model=64,
        pad_id=tokenizer.pad_id,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = pad_1d([tokenizer.encode(text, add_eos=True)], tokenizer.pad_id)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1)[0]
        pred_id = int(probs.argmax().item())

    print("输入文本:", text)
    print("预测类别:", id_to_label[pred_id])
    print("各类别概率:")
    for idx, prob in enumerate(probs.tolist()):
        print(f"  {id_to_label[idx]}: {prob:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="这个电影很好看")
    args = parser.parse_args()
    predict(args.text)


if __name__ == "__main__":
    main()

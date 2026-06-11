import argparse, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT_ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch, torch.nn.functional as F
torch.set_num_threads(1)  # CPU 教学 demo：避免某些机器上多线程过慢
from text_classification_task import TextClassificationModel
from utils.simple_tokenizer import CharTokenizer
from utils.training_utils import pad_1d, read_tsv, save_checkpoint

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--lr',type=float,default=1e-3); args=ap.parse_args()
    rows=read_tsv(PROJECT_ROOT/'data/classification/sentiment.tsv')
    texts=[r['text'] for r in rows]; labels=sorted({r['label'] for r in rows}); label_to_id={v:i for i,v in enumerate(labels)}
    tok=CharTokenizer().fit(texts); x=pad_1d([tok.encode(t,add_eos=True) for t in texts],tok.pad_id); y=torch.tensor([label_to_id[r['label']] for r in rows])
    model=TextClassificationModel(len(tok),len(label_to_id),d_model=64,pad_id=tok.pad_id); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    for e in range(1,args.epochs+1):
        logits=model(x); loss=F.cross_entropy(logits,y); opt.zero_grad(); loss.backward(); opt.step(); acc=(logits.argmax(-1)==y).float().mean().item(); print(f'epoch={e:02d} loss={loss.item():.4f} acc={acc:.3f}')
    out=PROJECT_ROOT/'run/text_classification'; tok.save(out/'tokenizer.json'); save_checkpoint(out/'model.pt',model,{'label_to_id':label_to_id}); print('saved to',out)
if __name__=='__main__': main()

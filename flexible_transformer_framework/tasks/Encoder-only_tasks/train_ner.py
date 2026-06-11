import argparse, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT_ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch, torch.nn.functional as F
torch.set_num_threads(1)  # CPU 教学 demo：避免某些机器上多线程过慢
from ner_task import NERModel
from utils.simple_tokenizer import CharTokenizer
from utils.training_utils import pad_1d, save_checkpoint

def read_conll(path):
    sents=[]; labs=[]; cs=[]; cl=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line:
            if cs: sents.append(cs); labs.append(cl); cs=[]; cl=[]
            continue
        a,b=line.split(); cs.append(a); cl.append(b)
    if cs: sents.append(cs); labs.append(cl)
    return sents,labs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--lr',type=float,default=1e-3); args=ap.parse_args()
    sents,labs=read_conll(PROJECT_ROOT/'data/ner/train.conll')
    tok=CharTokenizer().fit([''.join(s) for s in sents]); label_names=sorted({x for seq in labs for x in seq}); label_to_id={v:i for i,v in enumerate(label_names)}
    x=pad_1d([[tok.token_to_id.get(ch,tok.unk_id) for ch in s] for s in sents],tok.pad_id); y=pad_1d([[label_to_id[z] for z in seq] for seq in labs],-100)
    model=NERModel(len(tok),len(label_to_id),d_model=64,pad_id=tok.pad_id); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    for e in range(1,args.epochs+1):
        logits=model(x); loss=F.cross_entropy(logits.view(-1,len(label_to_id)),y.view(-1),ignore_index=-100); opt.zero_grad(); loss.backward(); opt.step()
        pred=logits.argmax(-1); valid=y!=-100; acc=(((pred==y)&valid).float().sum()/valid.float().sum()).item(); print(f'epoch={e:02d} loss={loss.item():.4f} token_acc={acc:.3f}')
    out=PROJECT_ROOT/'run/ner'; tok.save(out/'tokenizer.json'); save_checkpoint(out/'model.pt',model,{'label_to_id':label_to_id}); print('saved to',out)
if __name__=='__main__': main()

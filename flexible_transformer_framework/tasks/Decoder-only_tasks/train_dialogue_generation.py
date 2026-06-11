import argparse, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT_ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch, torch.nn.functional as F
torch.set_num_threads(1)  # CPU 教学 demo：避免某些机器上多线程过慢
from dialogue_generation_task import DialogueGenerationModel
from utils.simple_tokenizer import CharTokenizer
from utils.training_utils import pad_1d, read_tsv, save_checkpoint

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--lr',type=float,default=1e-3); args=ap.parse_args()
    rows=read_tsv(PROJECT_ROOT/'data/dialogue/dialogue.tsv'); tok=CharTokenizer().fit([r['context']+r['reply'] for r in rows])
    xs=[]; ys=[]
    for r in rows:
        prompt=tok.encode(r['context'],add_bos=True)+[tok.sep_id]; reply=tok.encode(r['reply'],add_eos=True); full=prompt+reply; inp=full[:-1]; lab=full[1:]
        for i in range(len(prompt)-1): lab[i]=-100
        xs.append(inp); ys.append(lab)
    x=pad_1d(xs,tok.pad_id); y=pad_1d(ys,-100); model=DialogueGenerationModel(len(tok),d_model=64,pad_id=tok.pad_id); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    for e in range(1,args.epochs+1):
        logits=model(x); loss=F.cross_entropy(logits.view(-1,len(tok)),y.view(-1),ignore_index=-100); opt.zero_grad(); loss.backward(); opt.step(); print(f'epoch={e:02d} loss={loss.item():.4f}')
    out=PROJECT_ROOT/'run/dialogue_generation'; tok.save(out/'tokenizer.json'); save_checkpoint(out/'model.pt',model); print('saved to',out)
if __name__=='__main__': main()

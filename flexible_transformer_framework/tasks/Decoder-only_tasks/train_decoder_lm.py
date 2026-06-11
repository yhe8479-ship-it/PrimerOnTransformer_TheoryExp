import argparse, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT_ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch, torch.nn.functional as F
torch.set_num_threads(1)  # CPU 教学 demo：避免某些机器上多线程过慢
from decoder_lm_task import DecoderOnlyLanguageModel
from utils.simple_tokenizer import CharTokenizer
from utils.training_utils import pad_1d, save_checkpoint

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--lr',type=float,default=1e-3); args=ap.parse_args()
    lines=[x.strip() for x in (PROJECT_ROOT/'data/language_model/lm.txt').read_text(encoding='utf-8').splitlines() if x.strip()]
    tok=CharTokenizer().fit(lines); enc=[tok.encode(x,add_bos=True,add_eos=True) for x in lines]; x=pad_1d([z[:-1] for z in enc],tok.pad_id); y=pad_1d([z[1:] for z in enc],-100)
    model=DecoderOnlyLanguageModel(len(tok),d_model=64,pad_id=tok.pad_id); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    for e in range(1,args.epochs+1):
        logits=model(x); loss=F.cross_entropy(logits.view(-1,len(tok)),y.view(-1),ignore_index=-100); opt.zero_grad(); loss.backward(); opt.step(); print(f'epoch={e:02d} loss={loss.item():.4f}')
    out=PROJECT_ROOT/'run/decoder_lm'; tok.save(out/'tokenizer.json'); save_checkpoint(out/'model.pt',model); print('saved to',out)
if __name__=='__main__': main()

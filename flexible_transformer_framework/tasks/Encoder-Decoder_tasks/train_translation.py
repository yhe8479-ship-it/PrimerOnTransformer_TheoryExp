import argparse, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT_ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch, torch.nn.functional as F
torch.set_num_threads(1)  # CPU 教学 demo：避免某些机器上多线程过慢
from seq2seq_translation_task import Seq2SeqTranslationModel
from utils.simple_tokenizer import CharTokenizer
from utils.training_utils import pad_1d, read_tsv, save_checkpoint

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=40); ap.add_argument('--lr',type=float,default=1e-3); args=ap.parse_args()
    rows=read_tsv(PROJECT_ROOT/'data/translation/zh_en.tsv'); src_texts=[r['zh'] for r in rows]; tgt_texts=[r['en'] for r in rows]
    st=CharTokenizer().fit(src_texts); tt=CharTokenizer().fit(tgt_texts); src=pad_1d([st.encode(x,add_eos=True) for x in src_texts],st.pad_id); enc=[tt.encode(x,add_bos=True,add_eos=True) for x in tgt_texts]; tgt=pad_1d([z[:-1] for z in enc],tt.pad_id); y=pad_1d([z[1:] for z in enc],-100)
    model=Seq2SeqTranslationModel(len(st),len(tt),d_model=64,pad_id=st.pad_id); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    for e in range(1,args.epochs+1):
        logits=model(src,tgt); loss=F.cross_entropy(logits.view(-1,len(tt)),y.view(-1),ignore_index=-100); opt.zero_grad(); loss.backward(); opt.step(); print(f'epoch={e:02d} loss={loss.item():.4f}')
    out=PROJECT_ROOT/'run/translation'; st.save(out/'src_tokenizer.json'); tt.save(out/'tgt_tokenizer.json'); save_checkpoint(out/'model.pt',model); print('saved to',out)
if __name__=='__main__': main()

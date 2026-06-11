
import json
from pathlib import Path

class CharTokenizer:
    PAD='<PAD>'; UNK='<UNK>'; BOS='<BOS>'; EOS='<EOS>'; SEP='<SEP>'
    def __init__(self, token_to_id=None):
        self.token_to_id = token_to_id or {self.PAD:0,self.UNK:1,self.BOS:2,self.EOS:3,self.SEP:4}
        self.id_to_token = {i:t for t,i in self.token_to_id.items()}
    @property
    def pad_id(self): return self.token_to_id[self.PAD]
    @property
    def unk_id(self): return self.token_to_id[self.UNK]
    @property
    def bos_id(self): return self.token_to_id[self.BOS]
    @property
    def eos_id(self): return self.token_to_id[self.EOS]
    @property
    def sep_id(self): return self.token_to_id[self.SEP]
    def __len__(self): return len(self.token_to_id)
    def fit(self, texts):
        for text in texts:
            for ch in text:
                if ch not in self.token_to_id:
                    idx=len(self.token_to_id); self.token_to_id[ch]=idx; self.id_to_token[idx]=ch
        return self
    def encode(self, text, add_bos=False, add_eos=False):
        ids=[]
        if add_bos: ids.append(self.bos_id)
        ids += [self.token_to_id.get(ch,self.unk_id) for ch in text]
        if add_eos: ids.append(self.eos_id)
        return ids
    def decode(self, ids, skip_special=True):
        specials={self.PAD,self.UNK,self.BOS,self.EOS,self.SEP}
        out=[]
        for i in ids:
            tok=self.id_to_token.get(int(i),self.UNK)
            if skip_special and tok in specials: continue
            out.append(tok)
        return ''.join(out)
    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(self.token_to_id,ensure_ascii=False,indent=2),encoding='utf-8')
    @classmethod
    def load(cls,path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

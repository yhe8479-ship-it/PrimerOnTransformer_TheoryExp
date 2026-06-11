
from pathlib import Path
import csv, torch

def pad_1d(sequences, pad_value=0):
    max_len=max(len(s) for s in sequences)
    out=torch.full((len(sequences),max_len),pad_value,dtype=torch.long)
    for i,s in enumerate(sequences): out[i,:len(s)]=torch.tensor(s,dtype=torch.long)
    return out

def read_tsv(path):
    with Path(path).open('r',encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f,delimiter='\t'))

def save_checkpoint(path, model, extra=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload={'model_state':model.state_dict()}
    if extra: payload.update(extra)
    torch.save(payload,path)

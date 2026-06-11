
from pathlib import Path
import runpy, sys
PROJECT_ROOT=Path(__file__).resolve().parent
TRAIN_FILES=[
 'tasks/Encoder-only_tasks/train_text_classification.py',
 'tasks/Encoder-only_tasks/train_ner.py',
 'tasks/Decoder-only_tasks/train_decoder_lm.py',
 'tasks/Decoder-only_tasks/train_dialogue_generation.py',
 'tasks/Encoder-Decoder_tasks/train_translation.py',
 'tasks/Encoder-Decoder_tasks/train_summarization.py',
]
if __name__=='__main__':
    for f in TRAIN_FILES:
        print('\n'+'='*80); print('Training:',f); print('='*80)
        old=sys.argv[:]; sys.argv=[f,'--epochs','3']; runpy.run_path(str(PROJECT_ROOT/f),run_name='__main__'); sys.argv=old

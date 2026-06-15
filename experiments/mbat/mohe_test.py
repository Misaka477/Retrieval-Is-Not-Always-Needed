"""Test MoHE 109M generation."""
import os, sys, torch
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'rina'))
sys.path.insert(0, os.path.join(BASE_DIR, 'archive'))
sys.path.insert(0, os.path.join(BASE_DIR, 'archive', 'models'))

os.environ['CUDA_HOME']='/home/aquama/miniconda3/envs/natalia'
os.environ['CPATH']='/home/aquama/miniconda3/envs/natalia/targets/x86_64-linux/include'
os.environ['LD_LIBRARY_PATH']='/home/aquama/miniconda3/envs/natalia/lib'
DEVICE='cuda'
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from model import MoHERWKV
from rwkv_v7_demo import RWKV_TOKENIZER

CKPT=os.path.join(BASE_DIR,'mohe_transferred_latest.pt')
TK=os.path.join(BASE_DIR,'checkpoints','rwkv_vocab_v20230424.txt')
VOCAB, DM, NP = 65536, 768, 1536

print('Loading MoHE 109M...')
model = MoHERWKV(VOCAB, DM, NP, n_experts=12, topk=2).to(DEVICE)
sd=torch.load(CKPT,map_location='cpu',weights_only=False)
sd=sd['model'] if 'model' in sd else sd
for k in list(sd.keys()):
    if k.startswith('prev_route') or '_batch_' in k: del sd[k]
model.load_state_dict(sd,strict=False)
model.eval()
tk=RWKV_TOKENIZER(TK)
print('Loaded.')

def gen(prompt,steps=50,temp=0.8):
    ids=tk.encode(prompt)
    x=torch.tensor([ids],dtype=torch.long,device=DEVICE)
    for _ in range(steps):
        logits,_=model(x)
        probs=torch.softmax(logits[:,-1].float()/temp,-1)
        probs[0,0]=0
        nxt=torch.multinomial(probs,1)
        x=torch.cat([x,nxt],1)
    return tk.decode(x[0].tolist())

prompts=["The capital of France is","User: What is 2+2?\n\nAssistant:"]
for p in prompts:
    print(f'PROMPT: {p}')
    print(f'GEN:    {gen(p, 50, 0.8)}')
    print()

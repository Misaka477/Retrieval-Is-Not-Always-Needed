import sys; sys.path.insert(0, 'D:/Software_Development/Project/RINA_Project')
import os; os.environ['RWKV_JIT_ON']='1'
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import torch
from rwkv.model import RWKV
from rina.rwkv_tokenizer import TRIE_TOKENIZER

m=RWKV(model='rwkv7-g1d-0.1b-20260129-ctx8192.pth',strategy='cuda fp32')
tok=TRIE_TOKENIZER('checkpoints/rwkv_vocab_v20230424.txt')
prompt='The Eiffel tower is in the city of'
p=torch.tensor([tok.encode(prompt)]).cuda()

g=p.clone()
with torch.no_grad():
    for _ in range(32):
        l,s=m.forward(g,None)
        probs=torch.softmax(l.float()/0.8,-1)
        next=torch.multinomial(probs,1)
        g=torch.cat([g,next],1)
print('OUTPUT:',repr(tok.decode(g[0].tolist()[len(tok.encode(prompt)):])))

"""Compare AR vs AR+StatefulDenoiser on diverse prompts."""
import sys; sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
import torch, os, time
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rwkv_v7_demo import RWKV, args, tokenizer, MODEL_PATH

DEVICE = 'cuda'; D = 768
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sd = torch.load(MODEL_PATH, map_location='cpu')
for k,v in list(sd.items()):
    if isinstance(v, torch.Tensor) and v.dtype != torch.float32: sd[k] = v.float()
m = RWKV(args).to(DEVICE); m.load_state_dict(sd, strict=False); m.eval()
for p in m.parameters(): p.requires_grad_(False)

class StatefulDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = D
        self.input_proj = nn.Linear(D*2, D)
        self.log_A = nn.Parameter(torch.zeros(D))
        self.B = nn.Linear(D, D, bias=False)
        self.C = nn.Linear(D, D, bias=False)
        self.gate = nn.Linear(D, D)
        self.out = nn.Linear(D, D)
    def forward(self, h, cond, state=None):
        B = h.shape[0]
        if state is None:
            state = h.new_zeros(B, self.d_model)
        inp = self.input_proj(torch.cat([h, cond], -1))
        A = torch.sigmoid(self.log_A)
        state = A * state + self.B(inp)
        h_out = h + torch.sigmoid(self.gate(inp)) * self.out(self.C(state))
        return h_out, state
    def reset_state(self, B=1):
        return torch.zeros(B, self.d_model, device=DEVICE)

dn = StatefulDenoiser().to(DEVICE)
dn.load_state_dict(torch.load(os.path.join(BASE_DIR, 'checkpoints', 'dn_stateful_final.pt'), weights_only=False)['dn'])
dn.eval()

def bb(x):
    pad = (16 - x.size(1)%16)%16
    if pad:
        xp = torch.cat([x, torch.zeros(1,pad,dtype=torch.long,device=DEVICE)],1)
        l,h = m(xp, return_h=True); return l[:,:x.size(1)], h[:,:x.size(1)]
    return m(x, return_h=True)

prompts = [
    "User: What is the capital of France?\n\nAssistant:",
    "User: Write a short poem about a cat.\n\nAssistant:",
    "User: Explain why the sky is blue.\n\nAssistant:",
    "User: What is 2+2?\n\nAssistant:",
    "The Eiffel tower is in the city of",
    "User: Who wrote Romeo and Juliet?\n\nAssistant:",
]

for prompt in prompts:
    p = torch.tensor([tokenizer.encode(prompt)]).to(DEVICE); plen = p.size(1)
    print(f'\nPrompt: {prompt}')
    for label, use_d in [('AR', False), ('AR+Dn', True)]:
        g = p.clone(); t0 = time.time()
        dn_state = dn.reset_state()
        with torch.no_grad():
            for _ in range(64):
                l, h = bb(g)
                if use_d:
                    cond = (torch.softmax(l*0.05,-1) @ m.head.weight).reshape(-1,D)
                    h_dn, dn_state = dn(h.reshape(-1,D), cond, dn_state)
                    l = m.head(h_dn.unsqueeze(0))
                probs = torch.softmax(l[:,-1].float()/0.8,-1); probs[:,0]=0
                g = torch.cat([g, torch.multinomial(probs,1)],1)
        print(f'  {label} ({time.time()-t0:.0f}s): {repr(tokenizer.decode(g[0].tolist()[plen:]))}')
print('\nDone.')

import sys,os; sys.path.insert(0,'.')
import torch, torch.nn.functional as F, statistics
from modules.cann_ssm import RINASeqModel
import modules.cann_ssm as _c; _c._setup_cuda_seq_v2=lambda:False

dm,V=768,4096; device='cuda'
st=torch.load('checkpoints/cann_lowrank_ep10.pt',map_location=device,weights_only=True)
m=RINASeqModel(V,d_model=dm,n_patterns=4096,beta=0.5,n_slots=V,attract_every=2,pattern_rank=128).to(device)
m.load_state_dict(st,strict=False); m.eval()

Wa=m.cell.gate_a.weight; ba=m.cell.gate_a.bias
Wb=m.cell.gate_b.weight; bb=m.cell.gate_b.bias
Wp=m.cell.proj_in.weight; bp=m.cell.proj_in.bias
Wn=m.cell.norm.weight; bn=m.cell.norm.bias
patterns=(m.cell.U@m.cell.V).detach()

def gate(h,x):
    c=torch.cat([h,x],dim=-1)
    a=torch.sigmoid(c@Wa.T+ba); b=torch.sigmoid(c@Wb.T+bb)
    return F.layer_norm(a*h+b*(x@Wp.T+bp),[dm],Wn,bn,eps=1e-5)
def attractor(h):
    s=(h@patterns.T)*0.5; a=torch.softmax(s,dim=-1)
    return h+0.1*(a@patterns-h)

torch.manual_seed(42)
seq=torch.randint(0,V,(8,256),device=device); emb=m.embed(seq)

# warmup: full attractor reference
h_ref=torch.zeros(8,dm,device=device)
for t in range(255):
    h_ref=F.layer_norm(attractor(gate(h_ref,emb[:,t,:])),[dm],Wn,bn,eps=1e-5)

print(f'{"Th":>6} {"Skip%":>7} {"cos_early":>10} {"cos_late":>10} {"min_cos":>10} {"Verdict":>15}')
print('-'*64)

for th in [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7]:
    h_skip=h_ref.clone()
    cos_vals=[]; skip_frac=[]
    for t in range(255):
        h_pred=h_skip.clone()
        h_g=gate(h_skip,emb[:,t,:])
        err=(h_g-h_pred).norm(dim=-1)/(h_pred.norm(dim=-1)+1e-8)
        do_att=err>th
        skip_frac.append((~do_att).float().mean().item())
        if do_att.any():
            h_a=F.layer_norm(attractor(h_g),[dm],Wn,bn,eps=1e-5)
            h_skip=torch.where(do_att.unsqueeze(-1),h_a,h_g)
        else:
            h_skip=h_g
        cos_vals.append(F.cosine_similarity(h_ref,h_skip,dim=-1).mean().item())
    sr=statistics.mean(skip_frac)
    ce=statistics.mean(cos_vals[:64])
    cl=statistics.mean(cos_vals[-64:])
    mc=min(cos_vals)
    v='STABLE' if mc>0.95 else 'MARGINAL' if mc>0.90 else 'DRIFTING'
    print(f'{th:6.2f} {sr*100:6.0f}% {ce:10.4f} {cl:10.4f} {mc:10.4f} {v:>15}')

"""REPR-ALIGN: teacher=clean, student=noisy embeddings, KL alignment."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import math, torch, torch.nn.functional as F, numpy as np
from tqdm import tqdm
from rina import MoHERWKV

device = 'cuda'
VOCAB, DM, NP = 65536, 768, 1536
BSZ, SEQ = 8, 512
LR = 1e-4; N_STEPS = 2000; SAVE_EVERY = 500
DATA_PATH = 'checkpoints/mohe_fw_rwkv_1b.npy'
CKPT_DIR = 'checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

# Teacher (frozen)
print("Loading teacher (frozen)...")
teacher = MoHERWKV(VOCAB, DM, NP, n_experts=12, topk=2).to(device)
sd = torch.load('checkpoints/mohe_transferred_latest.pt', map_location='cpu', weights_only=False)
s = sd.get('model', sd.get('model_state_dict', sd))
for k in list(s.keys()):
    if k.startswith('prev_route') or '_batch_' in k: del s[k]
teacher.load_state_dict(s, strict=False)
for p in teacher.parameters(): p.requires_grad = False
teacher.eval()

# Student (same init, trainable)
print("Creating student...")
student = MoHERWKV(VOCAB, DM, NP, n_experts=12, topk=2).to(device)
student.load_state_dict(teacher.state_dict(), strict=False)
opt = torch.optim.AdamW(student.parameters(), lr=LR)
print(f'Params: {sum(p.numel() for p in student.parameters())/1e6:.2f}M')

ids = torch.from_numpy(np.load(DATA_PATH, mmap_mode='r'))
nb = (len(ids) - 1) // (BSZ * SEQ)
perm = torch.randperm(nb)

student.train()
pbar = tqdm(range(N_STEPS))
for bi in pbar:
    s_idx = perm[bi % nb] * BSZ * SEQ
    x = ids[s_idx:s_idx + BSZ * SEQ].view(BSZ, SEQ).to(device, dtype=torch.long)
    opt.zero_grad()

    # Teacher: clean forward (target distribution)
    with torch.no_grad():
        clean_logits = teacher(x)
        if isinstance(clean_logits, tuple): clean_logits = clean_logits[0]

    # Student: forward with noisy embeddings
    noise_amp = max(0.05, 0.5 * (1 - bi / N_STEPS))  # anneal 0.5→0.05
    orig_emb = student.embed.weight.data.clone()
    student.embed.weight.data = orig_emb + torch.randn_like(orig_emb) * noise_amp

    student_logits = student(x)
    if isinstance(student_logits, tuple): student_logits = student_logits[0]

    student.embed.weight.data = orig_emb  # restore

    # Losses
    loss_ce = F.cross_entropy(student_logits.reshape(-1, VOCAB), x.reshape(-1), label_smoothing=0.1)
    loss_align = F.kl_div(
        F.log_softmax(student_logits / 0.1, dim=-1),
        F.softmax(clean_logits / 0.1, dim=-1).detach(),
        reduction='batchmean'
    )

    loss = loss_ce + 0.5 * loss_align
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0).item()
    opt.step()

    if bi % 200 == 0: torch.cuda.empty_cache()
    pbar.set_postfix(ce=f'{loss_ce.item():.1f}', align=f'{loss_align.item():.3f}', gn=f'{gn:.0f}')

    if bi % SAVE_EVERY == SAVE_EVERY - 1 or bi == N_STEPS - 1:
        torch.save({'model': student.state_dict(), 'opt': opt.state_dict()},
                   os.path.join(CKPT_DIR, 'mohe_repr_align.pt'))

print('Done → mohe_repr_align.pt')

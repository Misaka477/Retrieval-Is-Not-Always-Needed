"""Minimal audio world model: predict next audio frame in continuous space.

No tokenizer, no vocab, no CE. Pure next-frame MSE prediction.
If this works, WKV states naturally carry meaningful audio structure.
"""
import os, sys, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda'; DM = 128
SR, WIN, HOP = 16000, 512, 256  # 16kHz, 32ms window, 16ms hop

# ═══════════════════════════════════════════════════
# Synthetic audio: frequency sweeps + tones
# ═══════════════════════════════════════════════════

def generate_audio(n_seconds=120, sr=SR):
    """Generate frequency sweeps and tones."""
    t = np.linspace(0, n_seconds, int(sr * n_seconds), endpoint=False)
    audio = np.zeros_like(t)
    
    # Segment-based generation for structure
    seg_len = sr * 4  # 4-second segments
    for i in range(0, len(t), seg_len):
        seg_start = i // seg_len
        rng = np.random.RandomState(seg_start * 42)
        
        end = min(i + seg_len, len(t))
        seg_t = t[i:end] - t[i]
        seg = np.zeros_like(seg_t)
        
        choice = rng.randint(0, 4)
        if choice == 0:
            # Frequency sweep
            f_start = rng.uniform(80, 400)
            f_end = rng.uniform(150, 800)
            freq = f_start + (f_end - f_start) * seg_t / (seg_t[-1] + 1e-8)
            phase = 2 * math.pi * np.cumsum(freq) / sr
            seg = np.sin(phase)
        elif choice == 1:
            # Tone burst with silence
            burst = (seg_t % (sr // 2)) < (sr // 4)
            freq = rng.uniform(200, 600)
            tone = np.sin(2 * math.pi * freq * seg_t)
            seg[burst] = tone[burst] * 0.5
        elif choice == 2:
            # Chirp + noise
            f = rng.uniform(100, 500)
            freq = f + 200 * seg_t / (seg_t[-1] + 1e-8)
            phase = 2 * math.pi * np.cumsum(freq) / sr
            seg = np.sin(phase) + 0.2 * rng.randn(len(seg_t))
        else:
            # AM modulation
            freq = rng.uniform(200, 400)
            mod = rng.uniform(1, 5)
            seg = np.sin(2 * math.pi * freq * seg_t) * (0.5 + 0.5 * np.sin(2 * math.pi * mod * seg_t))
        
        seg = seg * 0.3 / (seg.std() + 1e-8)
        audio[i:end] = seg[:end-i]
    
    return torch.from_numpy(audio).float()

print('Generating audio...')
audio = generate_audio(60).to(DEVICE)  # 60 seconds
# Frame: overlapping windows
frames = audio.unfold(0, WIN, HOP)  # [n_frames, WIN]
n_frames = frames.size(0)
frames = frames - frames.mean(dim=-1, keepdim=True)
frames = frames / (frames.std(dim=-1, keepdim=True) + 1e-8)
print(f'Audio: {audio.size(0)} samples → {n_frames} frames ({n_frames * HOP / SR:.0f}s)')

# ═══════════════════════════════════════════════════
# Simple recurrence: predict next audio frame
# ═══════════════════════════════════════════════════

class AudioWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(WIN, DM)
        self.recur = nn.GRUCell(DM, DM)
        self.decoder = nn.Linear(DM, WIN)
        
    def forward(self, x):
        B, T, W = x.shape
        enc = self.encoder(x.view(-1, W)).view(B, T, DM)
        h = enc.new_zeros(B, DM)
        losses = []
        for t in range(T - 1):
            h = self.recur(enc[:, t], h)
            pred = self.decoder(h)
            loss = F.mse_loss(pred, x[:, t+1])
            losses.append(loss)
        return torch.stack(losses).mean()
    
    @torch.no_grad()
    def generate(self, n_steps=100):
        """Autoregressive generation."""
        self.eval()
        h = torch.zeros(1, DM, device=DEVICE)
        frame = torch.randn(1, WIN, device=DEVICE)
        frames_out = []
        for _ in range(n_steps):
            enc = self.encoder(frame)
            h = self.recur(enc, h)
            frame = self.decoder(h)
            frames_out.append(frame.cpu())
        return torch.cat(frames_out)

model = AudioWorldModel().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K')

# ═══════════════════════════════════════════════════
# Train: next-frame prediction, pure MSE
# ═══════════════════════════════════════════════════

BSZ, N_STEPS = 16, 5000
pbar = tqdm(range(N_STEPS))
for step in pbar:
    idx = torch.randint(0, n_frames - 64, (BSZ,))
    x = torch.stack([frames[i:i+64] for i in idx]).to(DEVICE)
    
    loss = model(x)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    
    if step % 1000 == 0:
        pbar.set_postfix(loss=f'{loss.item():.6f}')

print(f'Done. Final loss: {loss.item():.6f}')

# ═══════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════

print('\n=== Evaluating ===')
model.eval()
with torch.no_grad():
    # Test on held-out segment
    idx = torch.randint(0, n_frames - 64, (16,))
    x = torch.stack([frames[i:i+64] for i in idx]).to(DEVICE)
    
    # Compute per-step MSE on test batch
    h = model.encoder(x[:, 0])
    h = model.recur(h, torch.zeros(16, DM, device=DEVICE))
    pred = model.decoder(h)
    mse_step1 = F.mse_loss(pred, x[:, 1]).item()
    
    # Full autoregressive rollout on test
    h = torch.zeros(16, DM, device=DEVICE)
    errors = []
    for t in range(63):
        enc = model.encoder(x[:, t])
        h = model.recur(enc, h)
        pred = model.decoder(h)
        errors.append(F.mse_loss(pred, x[:, t+1]).item())
    
    print(f'  Step 1 MSE: {mse_step1:.6f}')
    print(f'  Mean MSE over 63 steps: {np.mean(errors):.6f}')
    print(f'  Error trajectory: {", ".join(f"{e:.4f}" for e in errors[::10])}')
    
    # Generate and check structure
    gen = model.generate(200)
    autocorr = (gen[0] * gen[0]).sum().item()  # just sanity
    print(f'  Generation: {gen.shape} frames, energy={autocorr:.2f}')
    print(f'  Frame energy var: {gen.var().item():.4f}')

torch.save({'model': model.state_dict()}, os.path.join(BASE_DIR := os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'audio_wm.pt'))
print('Saved.')

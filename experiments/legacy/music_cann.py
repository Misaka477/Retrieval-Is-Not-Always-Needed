"""Music World Model: WKV time + CANN frequency attractors.

WKV: processes audio frames in time → predicts next state
CANN: pulls the state toward learned attractor patterns (stabilizes frequency)
Decoder: projects stabilized state back to audio frame

Key hypothesis: CANN attractors naturally learn audio manifold (notes, chords, harmonics)
and keep the generated states on-manifold, reducing artifacts.
"""
import os, sys, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda'; DM = 128
SR, WIN, HOP = 16000, 1024, 256

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')

# ═══════════════════════════════════════════════════
# Music data
# ═══════════════════════════════════════════════════

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_freq(midi):
    return 440 * 2 ** ((midi - 69) / 12)

class MusicGenerator:
    SCALES = {'major': [0, 2, 4, 5, 7, 9, 11], 'minor': [0, 2, 3, 5, 7, 8, 10], 'pentatonic': [0, 2, 4, 7, 9]}

    def __init__(self, sr=SR):
        self.sr = sr
        self.rng = np.random.RandomState()

    def seed(self, s):
        self.rng = np.random.RandomState(s)

    def generate(self, duration=120):
        n_samples = int(self.sr * duration)
        audio = np.zeros(n_samples)
        key = 60 + self.rng.randint(-5, 5) * 12
        scale = self.rng.choice(['major', 'minor', 'pentatonic'])
        intervals = self.SCALES[scale]
        scale_notes = [key + i for i in intervals] + [key + 12 + i for i in intervals[:3]]
        t = np.arange(n_samples) / self.sr
        bpm = self.rng.randint(60, 140)
        bar_len = 60 / bpm * 4
        n_bars = int(duration / bar_len)

        for bar in range(n_bars):
            bar_start = bar * bar_len
            bar_end = min((bar + 1) * bar_len, duration)
            bs, be = int(bar_start * self.sr), int(bar_end * self.sr)
            if be <= bs: continue
            bar_t = t[bs:be] - bar_start

            chord_root = self.rng.choice(scale_notes)
            chord = [chord_root, chord_root + 4, chord_root + 7]
            if self.rng.random() > 0.5:
                chord.append(chord_root + 10)
            for note in chord:
                freq = midi_to_freq(note)
                vol = 0.12 / len(chord)
                audio[bs:be] += np.sin(2 * math.pi * freq * bar_t) * vol

            note_len = bar_len / (4 if self.rng.random() > 0.5 else 8)
            n_melody_notes = int(bar_len / note_len)
            for n_idx in range(n_melody_notes):
                ns = bs + int(n_idx * note_len * self.sr)
                ne = min(bs + int((n_idx + 1) * note_len * self.sr), be)
                if ne <= ns: continue
                mel_note = self.rng.choice(scale_notes) + 12
                freq = midi_to_freq(mel_note)
                n_t = np.arange(ne - ns) / self.sr
                att = min(int(0.01 * self.sr), ne - ns)
                rel = min(int(0.05 * self.sr), ne - ns)
                env = np.ones(ne - ns)
                env[:att] = np.linspace(0, 1, att)
                env[-rel:] = np.linspace(1, 0, rel)
                audio[ns:ne] += np.sin(2 * math.pi * freq * n_t) * env * 0.15

            for beat in range(4):
                bt = bar_start + beat * bar_len / 4
                if beat in (0, 2):
                    bs2, be2 = int(bt * self.sr), int(bt * self.sr + 0.05 * self.sr)
                    if be2 <= n_samples and be2 > bs2:
                        env = np.exp(-np.arange(be2 - bs2) / (0.01 * self.sr))
                        audio[bs2:be2] += np.sin(2 * math.pi * 60 * np.arange(be2 - bs2) / self.sr) * env * 0.3

        peak = np.abs(audio).max()
        return torch.from_numpy(audio / peak * 0.8 if peak > 0 else audio).float()


# ═══════════════════════════════════════════════════
# WKV + CANN model
# ═══════════════════════════════════════════════════

class WKVCell(nn.Module):
    """Simplified WKV: h_t = decay * h_{t-1} + k_t * v_t"""
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(DM, DM)
        self.value = nn.Linear(DM, DM)
        self.receptance = nn.Linear(DM, DM)
        self.decay = nn.Linear(DM, DM)
        nn.init.xavier_uniform_(self.key.weight, 0.5)
        nn.init.xavier_uniform_(self.value.weight, 0.5)
        nn.init.xavier_uniform_(self.receptance.weight, 0.5)

    def forward(self, x, h=None):
        k = torch.sigmoid(self.key(x))
        v = self.value(x)
        r = torch.sigmoid(self.receptance(x))
        if h is None:
            h = k * v
        else:
            decay = torch.sigmoid(self.decay(x))
            h = decay * h + k * v
        return r * (h / (h.norm(dim=-1, keepdim=True) + 1e-8))


class CANNLayer(nn.Module):
    """Continuous attractor: pull state toward nearest pattern.
    P = [p_1, ..., p_M] are learned attractor patterns.
    State update: h' = h + α * Σ_i overlap(h, p_i) * p_i
    Multiple iterations (N_STEPS) for convergence.
    """
    def __init__(self, n_patterns=64, n_steps=3):
        super().__init__()
        self.patterns = nn.Parameter(torch.randn(n_patterns, DM) * 0.1)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.n_steps = n_steps

    def forward(self, h):
        B, D = h.shape
        P = F.normalize(self.patterns, dim=-1)  # [M, DM]
        h_norm = F.normalize(h, dim=-1)  # [B, DM]

        for _ in range(self.n_steps):
            overlap = h_norm @ P.T  # [B, M] cosine similarity
            weights = torch.softmax(overlap * self.temperature, dim=-1)  # [B, M]
            # Pull toward patterns
            attract = weights @ P  # [B, DM]
            h_norm = F.normalize(h_norm + 0.5 * attract, dim=-1)

        # Scale back to original norm
        return h_norm * h.norm(dim=-1, keepdim=True)


class MusicCANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(WIN, DM)
        self.wkv = WKVCell()
        self.cann = CANNLayer(n_patterns=64, n_steps=3)
        self.decoder = nn.Linear(DM, WIN)
        self.norm = nn.LayerNorm(DM)

    def forward(self, x):
        B, T, W = x.shape
        enc = self.encoder(x.view(-1, W)).view(B, T, DM)
        h = None
        losses = []
        for t in range(T - 1):
            h = self.wkv(enc[:, t], h)
            h = self.cann(h)
            pred = self.decoder(h)
            loss = F.mse_loss(pred, x[:, t + 1])
            losses.append(loss)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def generate(self, seed_frames=4, n_steps=200):
        self.eval()
        idx = torch.randint(0, n_frames - seed_frames - n_steps, (1,))
        seed = frames[idx:idx+seed_frames].to(DEVICE).unsqueeze(0)

        h = None
        for t in range(seed_frames):
            enc = self.encoder(seed[:, t])
            h = self.wkv(enc, h)
            h = self.cann(h)

        gen = [seed[:, -1].cpu()]
        for _ in range(n_steps):
            enc = self.encoder(seed[:, -1])
            h = self.wkv(enc, h)
            h = self.cann(h)
            pred = self.decoder(h)
            gen.append(pred.cpu())
            seed = torch.cat([seed[:, 1:], pred.unsqueeze(0)], dim=1)

        return torch.cat(gen)


# ═══════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════

print('Generating music...')
mg = MusicGenerator()
all_audio = [mg.seed(s * 7 + 13) or mg.generate(8) for s in range(50)]
audio = torch.cat([mg.generate(8) for _ in range(50)]).to(DEVICE)
frames = audio.unfold(0, WIN, HOP)
n_frames = frames.size(0)
mu, std = frames.mean(dim=-1, keepdim=True), frames.std(dim=-1, keepdim=True) + 1e-8
frames = (frames - mu) / std
print(f'{audio.size(0)/SR:.0f}s audio, {n_frames} frames')

# ═══════════════════════════════════════════════════
# Train
# ═══════════════════════════════════════════════════

model = MusicCANN().to(DEVICE)
p_cann = sum(p.numel() for n, p in model.named_parameters() if 'cann' in n)
p_total = sum(p.numel() for p in model.parameters())
print(f'Params: {p_total/1e3:.1f}K  (CANN: {p_cann/1e3:.1f}K, patterns={model.cann.n_steps}×{((sum(p.numel() for p in model.cann.parameters())-1)/DM):.0f})')

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
BSZ, N_STEPS = 16, 5000
pbar = tqdm(range(N_STEPS))
t0 = time.time()

for step in pbar:
    idx = torch.randint(0, n_frames - 64, (BSZ,))
    x = torch.stack([frames[i:i+64] for i in idx]).to(DEVICE)
    loss = model(x)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    if step % 500 == 0:
        pbar.set_postfix(loss=f'{loss.item():.6f}')

print(f'Done in {(time.time()-t0)/60:.1f}min. Final loss: {loss.item():.6f}')

# ═══════════════════════════════════════════════════
# Eval
# ═══════════════════════════════════════════════════

print('\n=== Evaluation ===')
model.eval()
with torch.no_grad():
    idx = torch.randint(0, n_frames - 64, (16,))
    x = torch.stack([frames[i:i+64] for i in idx]).to(DEVICE)

    h = None
    errors = []
    for t in range(63):
        h = model.wkv(model.encoder(x[:, t]), h)
        h = model.cann(h)
        pred = model.decoder(h)
        errors.append(F.mse_loss(pred, x[:, t + 1]).item())

    e_mean = np.mean(errors)
    e_first = errors[0]
    e_last = np.mean(errors[-5:])

    # CANN pattern analysis
    patterns = F.normalize(model.cann.patterns, dim=-1)
    pat_sim = (patterns @ patterns.T).mean().item()

    print(f'  MSE: step1={e_first:.4f}  mean={e_mean:.4f}  last5={e_last:.4f}')
    print(f'  CANN pattern self-cos: {pat_sim:.3f} (1.0=all identical, 0=random)')
    print(f'  Error delta vs GRU: {"better" if e_mean < 0.097 else "worse"}')

torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'music_cann.pt'))
print('Saved.')

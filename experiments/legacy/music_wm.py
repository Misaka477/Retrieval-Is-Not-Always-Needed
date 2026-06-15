"""Music generation with pure MSE next-frame prediction.
Synthetic music: polyphonic notes, chords, rhythmic patterns.
No CE, no tokenizer, no vocab.
"""
import os, sys, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from tqdm import tqdm

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
DEVICE = 'cuda'; DM = 128
SR, WIN, HOP = 16000, 1024, 256  # 64ms window, 16ms hop for better frequency resolution

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')

# ═══════════════════════════════════════════════════
# Synthetic music generator
# ═══════════════════════════════════════════════════

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
BASE_FREQ = 261.63  # C4

def midi_to_freq(midi):
    """MIDI note number → frequency (A4=440Hz)."""
    return 440 * 2 ** ((midi - 69) / 12)

class MusicGenerator:
    """Generate polyphonic synthetic music with structure."""
    
    SCALES = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10],
        'pentatonic': [0, 2, 4, 7, 9],
    }
    
    def __init__(self, sr=SR):
        self.sr = sr
        self.rng = np.random.RandomState()
        
    def seed(self, s):
        self.rng = np.random.RandomState(s)
        
    def generate(self, duration=120):
        """Generate music with melody, chords, and rhythm."""
        n_samples = int(self.sr * duration)
        audio = np.zeros(n_samples)
        
        key = 60 + self.rng.randint(-5, 5) * 12  # C3-C5
        scale = self.rng.choice(['major', 'minor', 'pentatonic'])
        intervals = self.SCALES[scale]
        scale_notes = [key + i for i in intervals] + [key + 12 + i for i in intervals[:3]]
        
        t = np.arange(n_samples) / self.sr
        
        # Build chord progression
        bpm = self.rng.randint(60, 140)
        beats_per_bar = 4
        bar_len = 60 / bpm * beats_per_bar  # seconds per bar
        n_bars = int(duration / bar_len)
        
        for bar in range(n_bars):
            bar_start = bar * bar_len
            bar_end = min((bar + 1) * bar_len, duration)
            bs = int(bar_start * self.sr)
            be = int(bar_end * self.sr)
            
            if be <= bs:
                continue
                
            bar_t = t[bs:be] - bar_start
            
            # Chord (3-4 notes held across the bar)
            chord_root = self.rng.choice(scale_notes)
            if 'major' in scale or self.rng.random() > 0.3:
                chord = [chord_root, chord_root + 4, chord_root + 7]
            else:
                chord = [chord_root, chord_root + 3, chord_root + 7]
            if self.rng.random() > 0.5 and len(scale_notes) > 4:
                chord.append(chord_root + 10)  # 3rd octave
            
            for note in chord:
                freq = midi_to_freq(note)
                vol = 0.12 / len(chord)
                note_audio = np.sin(2 * math.pi * freq * bar_t) * vol
                # Add harmonics
                if self.rng.random() > 0.3:
                    note_audio += 0.3 * np.sin(2 * math.pi * freq * 2 * bar_t) * vol
                audio[bs:be] += note_audio
            
            # Melody (scattered notes, each lasting 1/8 or 1/4 bar)
            note_len = bar_len / (4 if self.rng.random() > 0.5 else 8)
            n_melody_notes = int(bar_len / note_len)
            
            melody_notes = []
            for n_idx in range(n_melody_notes):
                ns = bs + int(n_idx * note_len * self.sr)
                ne = min(bs + int((n_idx + 1) * note_len * self.sr), be)
                if ne <= ns:
                    continue
                
                # Pick melody note from scale, not necessarily chord tones
                mel_note = self.rng.choice(scale_notes) + 12  # one octave higher
                freq = midi_to_freq(mel_note)
                
                # Note envelope: attack, sustain, release
                n_t = np.arange(ne - ns) / self.sr
                attack = min(int(0.01 * self.sr), ne - ns)
                release = min(int(0.05 * self.sr), ne - ns)
                env = np.ones(ne - ns)
                env[:attack] = np.linspace(0, 1, attack)
                env[-release:] = np.linspace(1, 0, release)
                
                vol = 0.15
                note_audio = np.sin(2 * math.pi * freq * n_t) * env * vol
                audio[ns:ne] += note_audio
            
            # Percussion (simple kick on beat 1, snare on beat 3)
            for beat in range(beats_per_bar):
                beat_time = bar_start + beat * bar_len / beats_per_bar
                if beat == 0 or beat == 2:
                    b_start = int(beat_time * self.sr)
                    b_dur = int(0.05 * self.sr)
                    if b_start + b_dur <= n_samples:
                        env = np.exp(-np.arange(b_dur) / (0.01 * self.sr))
                        kick = np.sin(2 * math.pi * 60 * np.arange(b_dur) / self.sr) * env * 0.3
                        audio[b_start:b_start + b_dur] += kick
        
        # Normalize
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.8
        
        return torch.from_numpy(audio).float()


# ═══════════════════════════════════════════════════
# Model: next-frame prediction
# ═══════════════════════════════════════════════════

class MusicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(WIN, DM),
            nn.GELU(),
        )
        self.recur = nn.GRUCell(DM, DM)
        self.decoder = nn.Sequential(
            nn.Linear(DM, DM),
            nn.GELU(),
            nn.Linear(DM, WIN),
        )
        
    def forward(self, x):
        B, T, W = x.shape
        enc = self.encoder(x.view(-1, W)).view(B, T, DM)
        h = enc.new_zeros(B, DM)
        losses = []
        for t_ in range(T - 1):
            h = self.recur(enc[:, t_], h)
            pred = self.decoder(h)
            loss = F.mse_loss(pred, x[:, t_ + 1])
            losses.append(loss)
        return torch.stack(losses).mean()
    
    @torch.no_grad()
    def generate(self, seed_frames=4, n_steps=200, temp=1.0):
        self.eval()
        # Start from seed
        idx = torch.randint(0, n_frames - seed_frames - n_steps, (1,))
        seed = music_frames[idx:idx+seed_frames].to(DEVICE)
        
        h = music_model.encoder(seed[0:1])
        h = music_model.recur(h.squeeze(0), torch.zeros(1, DM, device=DEVICE))
        
        generated = [seed[0].cpu()]
        frame = seed[0:1]
        h = torch.zeros(1, DM, device=DEVICE)
        
        for t_ in range(seed_frames):
            enc = music_model.encoder(frame)
            h = music_model.recur(enc.squeeze(0), new_zeros if t_ > 0 else h)
        
        for t_ in range(seed_frames, n_steps):
            enc = music_model.encoder(frame)
            h = music_model.recur(enc.squeeze(0), h)
            pred = music_model.decoder(h)
            # Add small noise for diversity
            if temp > 0:
                pred = pred + torch.randn_like(pred) * temp * 0.01
            generated.append(pred.cpu())
        return torch.cat(generated)

# ═══════════════════════════════════════════════════
# Generate dataset
# ═══════════════════════════════════════════════════

print('Generating music dataset...')
mg = MusicGenerator()
all_audio = []
for s in range(50):  # 50 songs
    mg.seed(s * 7 + 13)
    all_audio.append(mg.generate(duration=6))
audio = torch.cat(all_audio).to(DEVICE)  # concatenate all songs
del all_audio

# Frame: overlapping windows
music_frames = audio.unfold(0, WIN, HOP)  # [n_frames, WIN]
n_frames = music_frames.size(0)

# Normalize frames
frame_mean = music_frames.mean(dim=-1, keepdim=True)
frame_std = music_frames.std(dim=-1, keepdim=True) + 1e-8
music_frames = (music_frames - frame_mean) / frame_std

print(f'Music: {audio.size(0)/SR:.0f}s, {n_frames} frames')

# ═══════════════════════════════════════════════════
# Train
# ═══════════════════════════════════════════════════

model = MusicModel().to(DEVICE)
print(f'Params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K')

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
BSZ, N_STEPS = 16, 5000
pbar = tqdm(range(N_STEPS))
t0 = time.time()

for step in pbar:
    idx = torch.randint(0, n_frames - 64, (BSZ,))
    x = torch.stack([music_frames[i:i+64] for i in idx]).to(DEVICE)
    
    loss = model(x)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    
    if step % 500 == 0:
        pbar.set_postfix(loss=f'{loss.item():.6f}')

print(f'Done in {(time.time()-t0)/60:.1f}min. Final loss: {loss.item():.6f}')

# ═══════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════

print('\n=== Evaluating ===')
model.eval()
with torch.no_grad():
    idx = torch.randint(0, n_frames - 64, (16,))
    x = torch.stack([music_frames[i:i+64] for i in idx]).to(DEVICE)
    
    h = model.encoder(x[:, 0])
    h = model.recur(h, torch.zeros(16, DM, device=DEVICE))
    pred = model.decoder(h)
    mse_1 = F.mse_loss(pred, x[:, 1]).item()
    
    # Rollout
    errors = []
    h = torch.zeros(16, DM, device=DEVICE)
    for t_ in range(63):
        enc = model.encoder(x[:, t_])
        h = model.recur(enc, h)
        pred = model.decoder(h)
        errors.append(F.mse_loss(pred, x[:, t_ + 1]).item())
    
    print(f'  Step 1 MSE: {mse_1:.6f}')
    print(f'  Mean MSE: {np.mean(errors):.6f}')
    print(f'  Steps 1-5: {", ".join(f"{e:.4f}" for e in errors[:5])}')
    
    # Generate continuation
    gen_start = torch.randint(0, n_frames - 100, (1,)).item()
    seed = music_frames[gen_start:gen_start+4].to(DEVICE).unsqueeze(0)
    
    h = model.encoder(seed[:, 0])
    h = model.recur(h, torch.zeros(1, DM, device=DEVICE))
    gen_frames = [seed[:, 0].cpu()]
    for t_ in range(1, 4):
        h = model.recur(model.encoder(seed[:, t_]), h)
    for _ in range(200):
        enc = model.encoder(seed[:, -1:])
        h = model.recur(enc.squeeze(0), h)
        pred = model.decoder(h)
        gen_frames.append(pred.cpu())
        seed = torch.cat([seed[:, 1:], pred.unsqueeze(0)], dim=1)
    
    gen_audio = torch.cat(gen_frames)
    print(f'  Generated {gen_audio.size(0)} frames ({(gen_audio.size(0) * HOP) / SR:.1f}s)')
    print(f'  Frame energy var: {gen_audio.var().item():.4f}')

torch.save({'model': model.state_dict()}, os.path.join(CKPT_DIR, 'music_wm.pt'))
print('\nSaved.')

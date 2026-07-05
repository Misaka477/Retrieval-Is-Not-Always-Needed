"""Python retention training: pure Python (80%) + DCLM English (20%) from SFT checkpoint."""
import os, argparse, torch, numpy as np
from tqdm import tqdm
from rina.model_jamba_qw3 import RINA_Jamba_QW3, QW3_Config

device = 'cuda'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_npy(path, target_tokens):
    data = np.load(path, mmap_mode='r')
    pos, seqs = 0, []
    while len(seqs) * 512 < target_tokens and pos + 513 <= len(data):
        seqs.append(data[pos:pos+513].tolist())
        pos += 512
    return seqs

def train(steps=50000, lr=3e-5, code_ratio=0.8):
    py_seqs  = load_npy(os.path.join(BASE, 'data/python_pretrain.npy'), 60_000_000)
    eng_seqs = load_npy(os.path.join(BASE, 'data/dclm_pretrain_llama_2000m.npy'), 15_000_000)
    print(f'Python: {len(py_seqs)} seqs, English: {len(eng_seqs)} seqs')

    out = os.path.join(BASE, 'models/out-jamba-py')
    os.makedirs(out, exist_ok=True)

    c = QW3_Config(vocab_size=128256,block_size=512,use_int4=False,
        n_embd=640,n_layer=16,n_head=10,n_kv_heads=5,d_c=160,head_dim=64,
        sparse_k=16,sparse_window=32,sparse_local_w=4,
        ssm_steps=3,quant_mode='q4k_q2v',ssm_qbits=0,weight_bits=16,embed_quant=False)
    model = RINA_Jamba_QW3(c).to(device)
    sd = torch.load(os.path.join(BASE, 'models/out-jamba-sft/sft_final.pt'),
                    map_location='cpu', weights_only=True)
    model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
    print('Loaded SFT checkpoint')

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    log_file = open(os.path.join(out, 'train.log'), 'w')
    pbar = tqdm(range(steps))
    gn_sum=0; gn_cnt=0; py_i=0

    for step in pbar:
        model.train()
        if np.random.random() < code_ratio:
            seq = py_seqs[py_i % len(py_seqs)]; py_i += 1
        else:
            seq = eng_seqs[np.random.randint(len(eng_seqs))]

        ids = torch.tensor(seq).long().to(device)
        x = ids[:-1].unsqueeze(0); y = ids[1:].unsqueeze(0)
        _, ce = model(x, y); ce.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        gn_sum+=gn.item(); gn_cnt+=1

        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                vs = py_seqs[(py_i) % len(py_seqs)]
                vx = torch.tensor(vs[:-1]).unsqueeze(0).to(device)
                vy = torch.tensor(vs[1:]).unsqueeze(0).to(device)
                _, vce = model(vx, vy)
            msg = f'step={step} ce={ce.item():.4f} val={vce.item():.4f} gn={gn_sum/max(gn_cnt,1):.2f}'
            log_file.write(msg+'\n'); log_file.flush()
            pbar.set_postfix(ce=f'{ce.item():.4f}', val=f'{vce.item():.4f}')

    torch.save({'model': model.state_dict(), 'step': steps}, f'{out}/py_final.pt')
    log_file.close()
    print('Done')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=50000)
    args = p.parse_args()
    train(steps=args.steps)

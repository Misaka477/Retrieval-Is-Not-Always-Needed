"""Step 2: Magicoder SFT (80%) + OASST dialog retention (20%)"""
import os, json, argparse, torch, numpy as np
from tqdm import tqdm
from rina.model_jamba_qw3 import RINA_Jamba_QW3, QW3_Config
from transformers import AutoTokenizer

device = 'cuda'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = AutoTokenizer.from_pretrained(
    os.path.join(BASE, 'models/teacher/LLM-Research/Llama-3___2-1B-Instruct'),
    local_files_only=True)

def load_magic(path, max_seq=512):
    seqs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            ids = tok.encode(d['text'])
            if 20 < len(ids) <= max_seq:
                seqs.append(ids)
    return seqs

def load_oasst(path, max_seq=300):
    from collections import defaultdict
    msgs = [json.loads(l) for l in open(path)]
    trees = defaultdict(list)
    for m in msgs:
        if m['lang'] != 'en': continue
        trees[m['message_tree_id']].append(m)
    seqs = []
    for tree in trees.values():
        root = next((m for m in tree if m['parent_id'] is None), None)
        if root is None: continue
        asst = next((m for m in tree if m.get('parent_id') == root['message_id'] and m['role'] == 'assistant'), None)
        if asst is None: continue
        text = f"User: {root['text']}\nAssistant: {asst['text']}"
        ids = tok.encode(text)
        if 20 < len(ids) <= max_seq:
            seqs.append(ids)
    return seqs

def train(steps=20000, lr=3e-5, code_ratio=0.8):
    magic_seqs = load_magic(os.path.join(BASE, 'data/magic_oss_en.jsonl'))
    oasst_seqs = load_oasst(os.path.join(BASE, 'data/oasst1/train.jsonl'))
    print(f'Magicoder: {len(magic_seqs)} | OASST: {len(oasst_seqs)}')

    c = QW3_Config(vocab_size=128256,block_size=512,use_int4=False,n_embd=640,
        n_layer=16,n_head=10,n_kv_heads=5,d_c=160,head_dim=64,
        sparse_k=16,sparse_window=32,sparse_local_w=4,
        ssm_steps=3,quant_mode='q4k_q2v',ssm_qbits=0,weight_bits=16,embed_quant=False)
    model = RINA_Jamba_QW3(c).to(device)
    sd = torch.load(os.path.join(BASE, 'models/out-jamba-py/py_final.pt'),
                    map_location='cpu', weights_only=True)
    model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
    print('Loaded Python-pretrained checkpoint')

    out = os.path.join(BASE, 'models/out-jamba-magic')
    os.makedirs(out, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    log_file = open(os.path.join(out, 'train.log'), 'w')
    pbar = tqdm(range(steps))
    gn_sum=gn_cnt=0

    for step in pbar:
        model.train()
        if np.random.random() < code_ratio:
            seq = magic_seqs[np.random.randint(len(magic_seqs))]
        else:
            seq = oasst_seqs[np.random.randint(len(oasst_seqs))]
        ids = torch.tensor(seq).long().to(device)
        _, ce = model(ids[:-1].unsqueeze(0), ids[1:].unsqueeze(0))
        ce.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        gn_sum+=gn.item(); gn_cnt+=1

        if step%200==0:
            model.eval()
            with torch.no_grad():
                vs = magic_seqs[np.random.randint(len(magic_seqs))]
                vx = torch.tensor(vs[:-1]).unsqueeze(0).to(device)
                vy = torch.tensor(vs[1:]).unsqueeze(0).to(device)
                _, vce = model(vx, vy)
            msg = f'step={step} ce={ce.item():.4f} val={vce.item():.4f} gn={gn_sum/max(gn_cnt,1):.2f}'
            log_file.write(msg+'\n'); log_file.flush()
            pbar.set_postfix(ce=f'{ce.item():.4f}', val=f'{vce.item():.4f}')

    torch.save({'model': model.state_dict(), 'step': steps}, f'{out}/magic_final.pt')
    log_file.close()
    print('Done')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=20000)
    args = p.parse_args()
    train(steps=args.steps)

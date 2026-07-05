"""Code SFT: Magicoder-OSS-Instruct + OASST dialog mixed training."""
import os, json, argparse, torch, numpy as np
from tqdm import tqdm
from rina.model_jamba_qw3 import RINA_Jamba_QW3, QW3_Config
from transformers import AutoTokenizer

device = 'cuda'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAGIC_PATH = os.path.join(BASE, 'data/magic_oss_en.jsonl')
OASST_PATH = os.path.join(BASE, 'data/oasst1/train.jsonl')
tok = AutoTokenizer.from_pretrained(
    os.path.join(BASE, 'models/teacher/LLM-Research/Llama-3___2-1B-Instruct'),
    local_files_only=True)

def prepare_magicoder(target=50000):
    """Download Magicoder-OSS-Instruct-75K and filter."""
    if os.path.exists(MAGIC_PATH):
        with open(MAGIC_PATH) as f:
            n = sum(1 for _ in f)
        print(f'Magicoder data exists: {n} examples')
        return
    
    print('Downloading Magicoder-OSS-Instruct-75K...')
    from datasets import load_dataset
    ds = load_dataset('ise-uiuc/Magicoder-OSS-Instruct-75K', split='train', trust_remote_code=True)
    
    with open(MAGIC_PATH, 'w') as f:
        count = 0
        for ex in ds:
            lang = ex.get('lang', '').lower()
            if lang != 'python': continue
            text = f"User: {ex['problem']}\nAssistant: {ex['solution']}"
            ids = tok.encode(text)
            if 20 < len(ids) <= 512:
                f.write(json.dumps({'text': text, 'tokens': len(ids)}) + '\n')
                count += 1
            if count >= target:
                break
    print(f'Saved {count} Python examples')

def load_data(ratio=0.8):
    """Load Magicoder + OASST, filter by length."""
    # Load Magicoder
    code_seqs = []
    with open(MAGIC_PATH) as f:
        for line in f:
            d = json.loads(line)
            code_seqs.append(tok.encode(d['text']))
    print(f'Code SFT: {len(code_seqs)} seqs')
    
    # Load OASST (dialog retention)
    oasst_seqs = []
    if os.path.exists(OASST_PATH):
        from collections import defaultdict
        msgs = [json.loads(l) for l in open(OASST_PATH)]
        # Group into conversations
        trees = defaultdict(list)
        for m in msgs:
            if m['lang'] != 'en': continue
            trees[m['message_tree_id']].append(m)
        for tid, tree in trees.items():
            # Get root → first assistant response
            root = next((m for m in tree if m['parent_id'] is None), None)
            if root is None: continue
            # Find assistant response
            asst = next((m for m in tree if m.get('parent_id') == root['message_id'] and m['role'] == 'assistant'), None)
            if asst is None: continue
            text = f"User: {root['text']}\nAssistant: {asst['text']}"
            ids = tok.encode(text)
            if 20 < len(ids) <= 300:
                oasst_seqs.append(ids)
    print(f'OASST: {len(oasst_seqs)} seqs')
    
    return code_seqs, oasst_seqs

def train(steps=20000, lr=3e-5, code_ratio=0.8):
    code_seqs, oasst_seqs = load_data()
    
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
    gn_sum=0; gn_cnt=0; code_i=0
    
    for step in pbar:
        model.train()
        if np.random.random() < code_ratio:
            seq = code_seqs[code_i % len(code_seqs)]
            code_i += 1
        else:
            seq = oasst_seqs[np.random.randint(len(oasst_seqs))]
        
        ids = torch.tensor(seq).long().to(device)
        x = ids[:-1].unsqueeze(0); y = ids[1:].unsqueeze(0)
        _, ce = model(x, y); ce.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        gn_sum+=gn.item(); gn_cnt+=1
        
        if step%200==0:
            model.eval()
            with torch.no_grad():
                vs = code_seqs[(code_i)%len(code_seqs)]
                vx = torch.tensor(vs[:-1]).unsqueeze(0).to(device)
                vy = torch.tensor(vs[1:]).unsqueeze(0).to(device)
                _, vce = model(vx, vy)
            msg = f'step={step} ce={ce.item():.4f} val={vce.item():.4f} gn={gn_sum/max(gn_cnt,1):.2f}'
            log_file.write(msg+'\n'); log_file.flush()
            pbar.set_postfix(ce=f'{ce.item():.4f}', val=f'{vce.item():.4f}')
    
    torch.save({'model': model.state_dict(), 'step': steps}, f'{out}/py_final.pt')
    log_file.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--download', action='store_true')
    p.add_argument('--steps', type=int, default=20000)
    args = p.parse_args()
    if args.download:
        prepare_magicoder()
    else:
        prepare_magicoder()
        train(steps=args.steps)

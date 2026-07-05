"""Proper SFT: loss only on completion (assistant response), not on prompt (user)."""
import os, json, argparse, torch, numpy as np
from tqdm import tqdm
from rina.model_jamba_qw3 import RINA_Jamba_QW3, QW3_Config
from transformers import AutoTokenizer

device = 'cuda'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tok = AutoTokenizer.from_pretrained(
    os.path.join(BASE, 'models/teacher/LLM-Research/Llama-3___2-1B-Instruct'),
    local_files_only=True)
IGNORE = -1

def load_data(path, max_seq=512):
    """Load and split prompt/completion. Filter to max_seq total len."""
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            text = d['text']  # "User: ...\nAssistant: ..."
            if '\nAssistant: ' not in text:
                prompt = text
                comp = ''
            else:
                prompt, comp = text.split('\nAssistant: ', 1)
                prompt += '\nAssistant: '
                comp = ' ' + comp  # space before first token for proper continuation
            
            # Tokenize
            p_ids = tok.encode(prompt)
            c_ids = tok.encode(comp) if comp else []
            total = len(p_ids) + len(c_ids)
            if 10 < total <= max_seq and len(c_ids) > 10:
                rows.append((p_ids, c_ids))
    return rows

def collate(p_ids, c_ids, max_seq=512):
    """Concatenate, mask prompt labels, truncate to max_seq."""
    labels = p_ids + c_ids
    input_ids = labels[:max_seq]
    labels = [IGNORE] * len(p_ids) + c_ids
    labels = labels[:max_seq]
    if len(input_ids) < max_seq:
        pad = [tok.pad_token_id or 0] * (max_seq - len(input_ids))
        input_ids += pad
        labels += [IGNORE] * (max_seq - len(labels))
    return input_ids, labels

def train(steps=20000, lr=3e-5):
    magic_path = os.path.join(BASE, 'data/magic_oss_en.jsonl')
    oasst_path = os.path.join(BASE, 'data/oasst1/train.jsonl')
    
    magic = load_data(magic_path)
    print(f'Magicoder: {len(magic)}')
    
    # Simple OASST: just take prompter→assistant pairs as prompt→completion
    from collections import defaultdict
    msgs = [json.loads(l) for l in open(oasst_path)]
    trees = defaultdict(list)
    for m in msgs:
        if m['lang'] != 'en': continue
        trees[m['message_tree_id']].append(m)
    oasst_rows = []
    for tree in trees.values():
        root = next((m for m in tree if m['parent_id'] is None), None)
        if not root: continue
        asst = next((m for m in tree if m.get('parent_id') == root['message_id'] and m['role'] == 'assistant'), None)
        if not asst: continue
        p = f"User: {root['text']}\nAssistant: "
        c = ' ' + asst['text']
        p_ids = tok.encode(p)
        c_ids = tok.encode(c)
        if 10 < len(p_ids) + len(c_ids) <= 300 and len(c_ids) > 10:
            oasst_rows.append((p_ids, c_ids))
    print(f'OASST: {len(oasst_rows)}')
    
    rows = magic + oasst_rows  # mix both
    
    c = QW3_Config(vocab_size=128256,block_size=512,use_int4=False,n_embd=640,
        n_layer=16,n_head=10,n_kv_heads=5,d_c=160,head_dim=64,
        sparse_k=16,sparse_window=32,sparse_local_w=4,
        ssm_steps=3,quant_mode='q4k_q2v',ssm_qbits=0,weight_bits=16,embed_quant=False)
    model = RINA_Jamba_QW3(c).to(device)
    sd = torch.load(os.path.join(BASE, 'models/out-jamba-py/py_final.pt'),
                    map_location='cpu', weights_only=True)
    model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
    print('Loaded Python-pretrained checkpoint')
    
    out = os.path.join(BASE, 'models/out-jamba-sft-v2')
    os.makedirs(out, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    log_file = open(os.path.join(out, 'train.log'), 'w')
    pbar = tqdm(range(steps))
    gn_sum = gn_cnt = 0
    
    for step in pbar:
        model.train()
        p_ids, c_ids = rows[np.random.randint(len(rows))]
        input_ids, labels = collate(p_ids, c_ids)
        x = torch.tensor(input_ids).long().to(device).unsqueeze(0)
        y = torch.tensor(labels).long().to(device).unsqueeze(0)
        if not torch.any(y != IGNORE): continue  # skip all-masked batch
        _, ce = model(x, y)
        ce.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        gn_sum += gn.item(); gn_cnt += 1
        
        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                p2, c2 = rows[np.random.randint(len(rows))]
                vx = torch.tensor(collate(p2, c2)[0]).unsqueeze(0).to(device)
                vy = torch.tensor(collate(p2, c2)[1]).unsqueeze(0).to(device)
                _, vce = model(vx, vy)
            msg = f'step={step} ce={ce.item():.4f} val={vce.item():.4f} gn={gn_sum/max(gn_cnt,1):.2f}'
            log_file.write(msg+'\n'); log_file.flush()
            pbar.set_postfix(ce=f'{ce.item():.4f}', val=f'{vce.item():.4f}')
    
    torch.save({'model': model.state_dict(), 'step': steps}, f'{out}/sft_final.pt')
    log_file.close()
    print('Done')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=20000)
    args = p.parse_args()
    train(steps=args.steps)

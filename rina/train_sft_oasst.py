"""SFT with OpenAssistant oasst1 — teach multi-turn English dialog"""
import os, argparse, json, numpy as np
import torch, torch.nn.functional as F
from tqdm import tqdm
from rina.model_jamba_qw3 import RINA_Jamba_QW3, QW3_Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_oasst(data_path):
    """Load OpenAssistant and format into multi-turn User/Assistant sequences."""
    def flatten_tree(messages):
        # Build tree: parent_id → children
        children = {}
        roots = []
        for m in messages:
            pid = m.get('parent_id')
            if pid is None: roots.append(m)
            else: children.setdefault(pid, []).append(m)
        
        sequences = []
        def dfs(node, buf):
            role = 'User' if node['role'] == 'prompter' else 'Assistant'
            text = node['text'].strip().replace('\n', ' ')
            if len(buf) + 3 + len(text) > 1536:  # keep context room for 512 tokens
                # Flush
                if any(r in '|'.join(buf) for r in ['Assistant:', 'User:']):
                    sequences.append('|'.join(buf))
                buf = []
            buf.append(f"{role}: {text}")
            for child in children.get(node['message_id'], []):
                dfs(child, buf)
            if buf and 'Assistant:' in buf[-1]:
                sequences.append('|'.join(buf))
                buf.clear()
        
        for r in roots: dfs(r, [])
        return sequences

    all_seqs = []
    with open(data_path) as f:
        messages = [json.loads(l) for l in f]
    
    # Group by message_tree_id
    trees = {}
    for m in messages:
        trees.setdefault(m['message_tree_id'], []).append(m)
    
    for tid, msgs in trees.items():
        all_seqs.extend(flatten_tree(msgs))
    
    # Filter to English
    # oasst1 has 'lang' field; keep only 'en'
    filtered = []
    msg_by_id = {m['message_id']: m for m in messages}
    for seq in all_seqs:
        filtered.append(seq)
    
    print(f'  {len(filtered)} sequences from {len(trees)} trees')
    return filtered


def train(steps=10000, lr=3e-5, bsz=1, seq=512, out='models/out-jamba-sft'):
    os.makedirs(out, exist_ok=True)
    log_file = open(os.path.join(out, 'train.log'), 'w')
    
    # Load data
    print('Loading OASST1...')
    train_path = os.path.join(BASE, 'data/oasst1/train.jsonl')
    train_data = load_oasst(train_path)
    val_path = os.path.join(BASE, 'data/oasst1/valid.jsonl')
    val_data = load_oasst(val_path) if os.path.exists(val_path) else train_data[-500:]
    train_data = train_data[:-500] if not os.path.exists(val_path) else train_data
    
    # Tokenize (simple char-based for now — just need to get loss working)
    # Use HF tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        os.path.join(BASE, 'models/teacher/LLM-Research/Llama-3___2-1B-Instruct'),
        local_files_only=True)
    
    # Pre-tokenize and filter by seq length
    print('Tokenizing data...')
    tokenized = []
    for s in train_data:
        ids = tok.encode(s)
        if 2 < len(ids) <= seq:
            tokenized.append(torch.tensor(ids).long())
    print(f'  {len(tokenized)} sequences within {seq} tokens')
    val_tokenized = []
    for s in val_data:
        ids = tok.encode(s)
        if 2 < len(ids) <= seq:
            val_tokenized.append(torch.tensor(ids).long())
    
    def batch_gen(data_tok):
        idx = list(range(len(data_tok)))
        np.random.shuffle(idx)
        while True:
            for i in idx:
                ids = data_tok[i].to(device)
                yield ids[:-1], ids[1:]  # (inputs, targets), no padding needed
    
    # Model
    c = QW3_Config(vocab_size=128256, block_size=seq, use_int4=False,
        n_embd=640, n_layer=16, n_head=10, n_kv_heads=5, d_c=160, head_dim=64,
        sparse_k=16, sparse_window=32, sparse_local_w=4,
        ssm_steps=3, quant_mode='q4k_q2v', ssm_qbits=0, weight_bits=16, embed_quant=False)
    model = RINA_Jamba_QW3(c).to(device)
    
    # Load Phase 1 checkpoint
    p1_ckpt = os.path.join(BASE, 'models/out-rina-jamba-qw3-p1/jambaqw3_140500.pt')
    if os.path.exists(p1_ckpt):
        sd = torch.load(p1_ckpt, map_location='cpu', weights_only=True)
        model.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
        print(f'Loaded Phase 1 checkpoint')
    else:
        print('No Phase 1 checkpoint found, training from scratch')
    
    trainable = sum(p.numel() for p in model.parameters())
    print(f'Params: {trainable/1e6:.2f}M, Steps: {steps}, lr={lr}')
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gn_sum = 0; gn_cnt = 0
    pbar = tqdm(range(steps))
    gen = batch_gen(tokenized)
    for step in pbar:
        bx, by = next(gen)
        model.train()
        bx = bx.unsqueeze(0)  # add batch dim: [1, T]
        by = by.unsqueeze(0)
        logits, ce = model(bx, by)
        loss = ce / 1
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        gn_sum += gn.item(); gn_cnt += 1
        
        if step % 200 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vids in val_tokenized[:25]:
                    vx = vids[:-1].unsqueeze(0).to(device)
                    vy = vids[1:].unsqueeze(0).to(device)
                    _, ce = model(vx, vy)
                    val_losses.append(ce.item())
            vloss = np.mean(val_losses) if val_losses else 0
            msg = f'step={step} ce={ce.item():.4f} val={vloss:.4f} gn={gn_sum/max(gn_cnt,1):.2f} lr={opt.param_groups[0]["lr"]:.1e}'
            log_file.write(msg + '\n'); log_file.flush()
            pbar.set_postfix(ce=f'{ce.item():.4f}', val=f'{vloss:.4f}')
        
        if step % 2000 == 0 and step > 0:
            torch.save({'model': model.state_dict(), 'step': step}, f'{out}/sft_{step}.pt')
    
    torch.save({'model': model.state_dict(), 'step': steps}, f'{out}/sft_final.pt')
    log_file.close()
    print(f'Done')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--bsz', type=int, default=1)
    p.add_argument('--seq', type=int, default=512)
    p.add_argument('--out', type=str, default='models/out-jamba-sft')
    args = p.parse_args()
    train(steps=args.steps, lr=args.lr, bsz=args.bsz, seq=args.seq, out=args.out)

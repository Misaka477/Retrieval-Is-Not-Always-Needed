"""Confidence Gate 实验: 模型"知不知道"能不能用熵来区分"""
import sys; sys.path.insert(0,'.')
import torch, math, numpy as np
from rina.model_a import RINA_A, RINA_A_Config
from transformers import GPT2Tokenizer
import torch.nn.functional as F

device='cuda'; torch.manual_seed(42)
tok=GPT2Tokenizer.from_pretrained('gpt2'); tok.pad_token=tok.eos_token

cfg=RINA_A_Config(vocab_size=50257,block_size=512,use_int4=True)
m=RINA_A(cfg).to(device); m.eval()
sd=torch.load('models/out-quant/rina-gen5-baseline-int4.pt',map_location=device,weights_only=False)['model']
m.load_state_dict(sd,strict=False)

# 测试几个句子, 看逐 token 的熵
tests = [
    'The capital of France is Paris. It is located in Europe.',
    'The cat sat on the mat and the dog ran in the park.',
    'Machine learning is a subset of artificial intelligence.',
]

all_entropies = []
all_correct_probs = []

for sent in tests:
    ids=tok.encode(sent)[:32]
    x=torch.tensor(ids,device=device).unsqueeze(0)
    with torch.no_grad():
        out=m(x[:,:-1],x[:,1:])
        l=out[0]  # [B, T-1, V]
    probs=F.softmax(l[0],-1)   # [T-1, V]
    ent=-(probs*probs.log()).sum(-1)  # [T-1]
    correct=probs[range(len(ids)-1), x[0,1:]]
    
    print(f'\n=== {sent[:40]}... ===')
    for t in range(min(len(ids)-1, 20)):
        token=tok.decode([ids[t]]).strip()
        next_tok=tok.decode([ids[t+1]]).strip()
        all_entropies.append(ent[t].item())
        all_correct_probs.append(correct[t].item())
        if t < 15:
            print(f'  {token:>10s} → {next_tok:<10s}  ent={ent[t]:.2f}  correct_p={correct[t]*100:.1f}%')

# 分析: 高熵的 token 有什么特征
all_entropies=np.array(all_entropies)
all_correct_probs=np.array(all_correct_probs)
print('\n=== Entropy analysis ===')
for thresh in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
    mask=all_entropies>thresh
    if mask.sum()>0:
        print(f'  entropy > {thresh:.1f}: {mask.sum()} tokens, avg correct prob={all_correct_probs[mask].mean()*100:.1f}%')
    else:
        print(f'  entropy > {thresh:.1f}: 0 tokens')

# 最高/最低熵的 token
sort_idx=np.argsort(all_entropies)
print(f'\n  Lowest entropy tokens (most confident):')
for i in sort_idx[:5]:
    print(f'    ent={all_entropies[i]:.3f} correct_p={all_correct_probs[i]*100:.1f}%')
print(f'  Highest entropy tokens (least confident):')
for i in sort_idx[-5:]:
    print(f'    ent={all_entropies[i]:.3f} correct_p={all_correct_probs[i]*100:.1f}%')

# 用 ROM 做概率修正实验
# 选一个 low-confidence token, 注入 ROM 看它是否能被修正
print('\n=== ROM injection on low-confidence token ===')
# 找一个模型不太确定的 token
low_conf_idx=sort_idx[0]  # start from lowest for reference
# Actually, find a token with medium-to-high entropy
for idx in reversed(sort_idx[-20:]):
    # Reconstruct its context
    sent_idx = 0
    pos = idx
    for si, sent in enumerate(tests):
        ids=tok.encode(sent)[:32]
        if pos < len(ids)-1:
            context=ids[:pos+1]
            correct_id=ids[pos+1]
            correct_word=tok.decode([correct_id]).strip()
            prompt=tok.decode(context)
            break
        pos -= (len(ids)-1)
    
    if correct_id is None or all_entropies[idx] < 2.0:
        continue
    
    actual_entropy=all_entropies[idx]
    actual_correct=all_correct_probs[idx]
    print(f'  Token: {correct_word!r} (id={correct_id}), ent={actual_entropy:.2f}, base_p={actual_correct*100:.1f}%')
    
    # Now create a ROM that "teaches" the model this specific fact
    rom_text=f'This is a text about {correct_word}. The word {correct_word} is very important here.'
    rom_ids=tok.encode(rom_text)[:32]
    rx=torch.tensor(rom_ids,device=device).unsqueeze(0)
    with torch.no_grad():
        _,_,rlats=m(rx,rx)
    rom_lats=rlats[0,5].detach()
    d=m.transformer.h[5].attn
    cq=rom_lats.unsqueeze(0)
    with torch.no_grad():
        k_rom=d.w_uk(cq).view(1,-1,d.n_kv,d.d_h).transpose(1,2)
        v_rom=d.w_k2v(d.w_uk(cq)).view(1,-1,d.n_kv,d.d_h).transpose(1,2)
        if d.n_rep>1:
            k_rom=k_rom.repeat_interleave(d.n_rep,1)
            v_rom=v_rom.repeat_interleave(d.n_rep,1)
    
    # Forward with ROM injection
    ctx_ids=context
    ctx=torch.tensor(ctx_ids,device=device).unsqueeze(0)
    B,T=ctx.shape
    h=m.transformer.drop(m.transformer.wte(ctx))
    for li,block in enumerate(m.transformer.h):
        bb=block.attn
        ln=block.ln1(h)
        cq_=bb.q_norm(bb.w_dqkv(ln))
        qc=bb.w_uq(cq_).view(B,T,bb.n_head,bb.d_h).transpose(1,2)
        kc=bb.w_uk(cq_).view(B,T,bb.n_kv,bb.d_h).transpose(1,2)
        v=bb.w_k2v(bb.w_uk(cq_)).view(B,T,bb.n_kv,bb.d_h).transpose(1,2)
        qr=bb.w_qr(ln).view(B,T,bb.n_head,bb.d_hr).transpose(1,2)
        kr=bb.w_kr(ln).view(B,T,bb.n_kv,bb.d_hr).transpose(1,2)
        qr,kr=bb.rope_q(qr),bb.rope(kr)
        if bb.n_rep>1:
            kc=kc.repeat_interleave(bb.n_rep,1);v=v.repeat_interleave(bb.n_rep,1);kr=kr.repeat_interleave(bb.n_rep,1)
        if 4<=li<=7:
            kr_rom=torch.zeros(1,bb.n_head,rom_lats.shape[0],bb.d_hr,device=x.device)
            kc=torch.cat([k_rom,kc],dim=2);v=torch.cat([v_rom,v],dim=2);kr=torch.cat([kr_rom,kr],dim=2)
        nk=kc.size(2)
        q_=torch.cat([qc,qr],-1);k_=torch.cat([kc,kr],-1)
        a=(q_@k_.transpose(-2,-1))/math.sqrt(q_.size(-1))
        mask=torch.triu(torch.full((T,nk),float('-inf'),device=x.device),diagonal=1)
        a=a+mask.unsqueeze(0).unsqueeze(0);a=F.softmax(a,-1);y_=a@v
        y_=y_.transpose(1,2).contiguous().view(B,T,-1)
        h=h+bb.resid_drop(bb.c_proj(y_))+block.mlp(block.ln2(h))
    h=m.transformer.ln_f(h)
    logits=m.lm_head(h[:,-1:])
    p_rom=F.softmax(logits[0,-1],-1)[correct_id].item()
    
    delta=(p_rom-actual_correct)*100
    print(f'    With ROM: p={p_rom*100:.1f}%  (delta={delta:+.1f}%)')
    break  # just one sample for now

# MoHE-RWKV v7 — Architecture & Training

## Config (109.44M)

| Param | Value |
|-------|-------|
| dm | 768 |
| NP (patterns) | 1536 |
| n_experts | 12 |
| topk | 2 |
| route_noise | 0.0 |
| aux_loss_weight | 0.1 |
| route_raw | ×3.0 |
| router_bias | N(0, 0.5²) |
| entropy_bonus | -0.01 × H |
| depth chain | h = h_new |
| vocab | 65536 (RWKV) |
| head_size | 64, H=12 |
| SEQ | 512 |
| BSZ | 4 |

## Dataflow

```
x: [B,S] token IDs
  │
  ▼
Embedding(65536→768) + LayerNorm  ← weight tied with head
  │ emb: [B,S,768]
  ▼
RWKV-v7 Time Mixing (CUDA kernel, float32)
  r=Linear(emb)  k=Linear(emb)  v=Linear(emb)  a=Linear(emb) * 0.01
  w=exp(-exp(tmix_w))  → per-head exponential decay
  h = WKV7(r,w,k,v,-a,b)  → [B,S,768]
  │
  ▼
Depth Loop (×3, h = h_new chain):
  ① Router: Linear(h) + bias → [B,T,12]  (per-token routing)
     ×3.0 scaling → entropy_bonus(-0.01×H) in aux loss
     no noise, no inertia smoothing

  ② Attractor ×12 (batched, [12,B,T,768]):
     P = patternsᵀ @ patterns
     field = h @ P → Proj(Linear→GELU→Linear)
     field = FieldMix + LayerNorm
     gate = sigmoid(Linear(cat(h, emb)))
     out = h + gate · field          (×1.0, no attenuation)

  ③ ExpertNorm → topk=2 mask (per-token) → Consolidate
     cat([B,T,12,768]) → Linear(9216→768) → LayerNorm

  │ h_new: [B,T,768] → h = h_new (chain to next depth)
  ▼
Head Linear(768→65536)  ← tied to Embedding, bias=-10.8
  │ logits: [B,T,65536]
  ▼
Loss = CE(logits, x) + aux_loss

Auxiliary Loss (train only):
  al = 0.1 × 12 × mean(f × p)           load balancing
  zl = 1e-4 × mean(logsumexp²)          confidence
  rl = 1e-3 × mean(route²)              regularization
  eb = -0.01 × H                         entropy bonus
```

## Training Recipe

### Data (500M tokens)

| Source | Tokens | Ratio |
|--------|--------|-------|
| FineWeb (English) | 250M | 50% |
| StarCoder (code) | 100M | 20% |
| OpenWebMath | 75M | 15% |
| SkyPile-150B (Chinese) | 75M | 15% |

- Tokenizer: RWKV trie (`rwkv_vocab_v20230424`)
- Shuffle interleave, streaming from HuggingFace
- 1% held out as validation set (~5M tokens)

### Training

| Param | Value |
|-------|-------|
| Optimizer | AdamW |
| LR | 3e-4 → cosine → 3e-5 |
| Warmup | 200 steps |
| N_STEPS | 90000 |
| Save interval | 1500 steps |
| Eval interval | 500 steps (100 batches) |
| Init | RWKV-7 transferred (`BlinkDL/rwkv7-g1`) |
| Hardware | RTX 3070 Ti 8GB |
| Speed | ~2 it/s @ SEQ=512 (4×1024 tok/s) |
| Total time | ~12.5 hours for 90000 steps |

### Performance (step 72000 / 90000)

| Metric | Value |
|--------|-------|
| Train ppl | 4.9 |
| Val ppl | 4.3 |
| Loss | 1.58 |
| Grad norm | 2.4 |
| Route entropy | 0.5 (differentiated) |
| Exp sim | 0.0000 (not diverged) |
| Aux loss | 0.13 |

### Key Findings

1. **Per-token routing is essential** — `.mean(1)` in router kills differentiation
2. **topk=2 forces sparse specialization** — ent drops from 2.48 to 0.2-0.5
3. **×1.0 expert scale** — removes bottleneck from residual attenuation
4. **Depth chain fixes** — `h = h_new` makes depth=3 meaningful (DoT-like)
5. **No cross-step smoothing** — `prev_route`/`inertia` not needed with per-token routing
6. **ent=0.2-0.5 is ideal** — 2-3 active experts per token, below 1.0

"""Latent ROM — 用户可读写的 MLA latent 知识库"""
import torch
import faiss
import numpy as np
from typing import Optional, List, Tuple

class LatentROM:
    """
    MLA latent 知识库。
    写入: 文本 → tokenize → RINA 前向 → c_kv → FAISS 索引
    读取: c_q → FAISS search  → top-K latent → 投影为 K/V → 注入 attention
    """
    def __init__(self, dim: int = 128, use_gpu: bool = False):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # 内积搜索（=cosine 相似度，归一化后）
        self.texts: List[str] = []           # 对应的原始文本
        self.latents: List[torch.Tensor] = [] # 对应的 c_kv latent

    def write(self, latents: torch.Tensor, texts: Optional[List[str]] = None):
        """写入 latents 到索引。
        latents: [T, d_c] — 一个句子的 per-token latent
        texts: [T] — 对应的 token 文本（可选，用于 debug）
        """
        lats = latents.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(lats)  # 归一化到单位向量
        self.index.add(lats)
        self.latents.append(latents.cpu())
        if texts:
            self.texts.extend(texts)

    def search(self, c_q: torch.Tensor, k: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """检索与 c_q 最相似的 k 个 latent。
        c_q: [d_c] — query latent
        返回: (top_k_latents, scores) — [k, d_c], [k]
        """
        if self.index.ntotal == 0:
            return torch.zeros(0, self.dim), torch.zeros(0)
        q = c_q.cpu().numpy().astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, idx = self.index.search(q, min(k, self.index.ntotal))
        idx = idx[0]
        scores = scores[0]
        # collect from flat index
        tl = []
        for i in idx:
            offset = 0
            for lat in self.latents:
                if offset + len(lat) > i:
                    tl.append(lat[i - offset])
                    break
                offset += len(lat)
        if not tl:
            return torch.zeros(0, self.dim), torch.zeros(0)
        return torch.stack(tl).to(c_q.device), torch.tensor(scores).to(c_q.device)

    def get_kv(self, model, layer_idx: int, c_q: torch.Tensor, k: int = 8):
        """检索 top-K latents 并投影为 K/V 用于 attention。
        返回: (k_rom, v_rom) — [1, n_head, K, d_h], [1, n_head, K, d_h]
        """
        lats, scores = self.search(c_q, k)
        if len(lats) == 0:
            return None, None
        # 用指定层的 MLA 投影矩阵将 latent 映射为 K/V
        block = model.transformer.h[layer_idx].l2
        cq = lats.unsqueeze(0)  # [1, K, d_c]
        with torch.no_grad():
            kc = block.w_uk(cq).view(1, -1, block.n_kv, block.d_h).transpose(1, 2)
            v = block.w_k2v(block.w_uk(cq)).view(1, -1, block.n_kv, block.d_h).transpose(1, 2)
            if block.n_rep > 1:
                kc = kc.repeat_interleave(block.n_rep, 1)
                v = v.repeat_interleave(block.n_rep, 1)
        return kc, v

    def write_text(self, model, tokenizer, text: str, layer_idx: int = 5, max_len: int = 128):
        """便捷方法: 直接写入文本"""
        import torch.nn.functional as F
        ids = tokenizer.encode(text)[:max_len]
        x = torch.tensor([ids])
        if next(model.parameters()).is_cuda:
            x = x.cuda()
        with torch.no_grad():
            from rina.model_cf import RINA_CF
            if isinstance(model, RINA_CF):
                h = model.transformer.drop(model.transformer.wte(x))
                for li, blk in enumerate(model.transformer.h):
                    ln = blk.ln1(h)
                    if li == layer_idx:
                        cq = blk.l2.q_norm(blk.l2.w_dqkv(ln))
                        break
                    out, _ = blk.l2(ln)
                    h = h + out + blk.mlp(blk.ln2(h))
            else:
                _, _, lats = model(x, x)
                cq = lats[0, layer_idx]
        texts = [tokenizer.decode([i]) for i in ids]
        # cq shape could be [B, T, d_c] or [T, d_c], flatten to [T, d_c]
        if cq.dim() == 3:
            cq = cq[0]
        self.write(cq, texts)
        return len(ids)

    def __len__(self):
        return self.index.ntotal

    def clear(self):
        self.index = faiss.IndexFlatIP(self.dim)
        self.texts.clear()
        self.latents.clear()

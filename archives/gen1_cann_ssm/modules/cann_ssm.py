"""
CANN-SSM Cell — RINA 核心引擎. 含 JIT + 稀疏吸引子优化.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import ctypes, os


# ── CUDA kernel wrapper (for forward only) ──
_cuda_dll = None
def _get_cuda_dll():
    global _cuda_dll
    if _cuda_dll is None:
        dll_path = os.path.join(os.path.dirname(__file__), "cann_step.dll")
        if os.path.exists(dll_path):
            _cuda_dll = ctypes.CDLL(dll_path)
            _cuda_dll.launch_cann_step.restype = None
            _cuda_dll.launch_cann_step.argtypes = [ctypes.c_void_p] * 14 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]
    return _cuda_dll


# ── CUDA sequence v2 Dll setup (forward+backward for training) ──
_cuda_seq_v2_ready = False
def _setup_cuda_seq_v2():
    global _cuda_seq_v2_ready
    if _cuda_seq_v2_ready:
        return True
    dll = _get_cuda_dll()
    if dll is None:
        return False
    try:
        dll.launch_cann_sequence_v2.restype = None
        dll.launch_cann_sequence_v2.argtypes = (
            [ctypes.c_void_p] * 30 + [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_int]
        )
        dll.launch_cann_sequence_backward.restype = None
        dll.launch_cann_sequence_backward.argtypes = (
            [ctypes.c_void_p] * 50 + [ctypes.c_int] * 5 + [ctypes.c_float, ctypes.c_int]
        )
        _cuda_seq_v2_ready = True
        return True
    except AttributeError:
        return False


class CANNSequenceCUDA(torch.autograd.Function):
    """Full-sequence CANN forward+backward via CUDA kernels for training."""

    @staticmethod
    def forward(ctx, h_init, emb, x_tokens, patterns, slot_table,
                wa, ba, wb, bb, wg, bg, wp, bp, wn, bn,
                sn_w, sn_b, head_w, head_b, beta, attract_every):
        dll = _get_cuda_dll()
        _setup_cuda_seq_v2()
        bs, seq, dm = emb.shape
        np_ = patterns.size(0)
        vs = head_w.size(0)

        logits = torch.zeros(bs, seq, vs, device=emb.device)
        saved_h_ssm = torch.zeros(bs, seq, dm, device=emb.device)
        saved_gate_a = torch.zeros(bs, seq, dm, device=emb.device)
        saved_gate_b = torch.zeros(bs, seq, dm, device=emb.device)
        saved_alpha = torch.zeros(bs, seq, dm, device=emb.device)
        saved_attn = torch.zeros(bs, seq, np_, device=emb.device)
        saved_h_new = torch.zeros(bs, seq, dm, device=emb.device)
        saved_cmean = torch.zeros(bs, seq, device=emb.device)
        saved_cinv_std = torch.zeros(bs, seq, device=emb.device)
        saved_hmean = torch.zeros(bs, seq, device=emb.device)
        saved_hinv_std = torch.zeros(bs, seq, device=emb.device)

        dll.launch_cann_sequence_v2(
            ctypes.c_void_p(h_init.data_ptr()),
            ctypes.c_void_p(emb.data_ptr()),
            ctypes.c_void_p(x_tokens.data_ptr()),
            ctypes.c_void_p(patterns.data_ptr()),
            ctypes.c_void_p(slot_table.data_ptr()),
            ctypes.c_void_p(wa.data_ptr()), ctypes.c_void_p(ba.data_ptr()),
            ctypes.c_void_p(wb.data_ptr()), ctypes.c_void_p(bb.data_ptr()),
            ctypes.c_void_p(wg.data_ptr()), ctypes.c_void_p(bg.data_ptr()),
            ctypes.c_void_p(wp.data_ptr()), ctypes.c_void_p(bp.data_ptr()),
            ctypes.c_void_p(wn.data_ptr()), ctypes.c_void_p(bn.data_ptr()),
            ctypes.c_void_p(sn_w.data_ptr()), ctypes.c_void_p(sn_b.data_ptr()),
            ctypes.c_void_p(head_w.data_ptr()), ctypes.c_void_p(head_b.data_ptr()),
            ctypes.c_void_p(logits.data_ptr()),
            ctypes.c_void_p(saved_h_ssm.data_ptr()),
            ctypes.c_void_p(saved_gate_a.data_ptr()),
            ctypes.c_void_p(saved_gate_b.data_ptr()),
            ctypes.c_void_p(saved_alpha.data_ptr()),
            ctypes.c_void_p(saved_attn.data_ptr()),
            ctypes.c_void_p(saved_h_new.data_ptr()),
            ctypes.c_void_p(saved_cmean.data_ptr()),
            ctypes.c_void_p(saved_cinv_std.data_ptr()),
            ctypes.c_void_p(saved_hmean.data_ptr()),
            ctypes.c_void_p(saved_hinv_std.data_ptr()),
            bs, seq, dm, np_, vs,
            ctypes.c_float(beta),
            attract_every,
        )

        ctx.save_for_backward(
            h_init, emb, x_tokens, patterns, slot_table,
            wa, ba, wb, bb, wg, bg, wp, bp, wn, bn,
            sn_w, sn_b, head_w, head_b,
            saved_h_ssm, saved_gate_a, saved_gate_b,
            saved_alpha, saved_attn, saved_h_new,
            saved_cmean, saved_cinv_std,
            saved_hmean, saved_hinv_std,
        )
        ctx.attract_every = attract_every
        ctx.beta = beta
        return logits

    @staticmethod
    def backward(ctx, grad_logits):
        (h_init, emb, x_tokens, patterns, slot_table,
         wa, ba, wb, bb, wg, bg, wp, bp, wn, bn,
         sn_w, sn_b, head_w, head_b,
         saved_h_ssm, saved_gate_a, saved_gate_b,
         saved_alpha, saved_attn, saved_h_new,
         saved_cmean, saved_cinv_std,
         saved_hmean, saved_hinv_std) = ctx.saved_tensors

        dll = _get_cuda_dll()
        bs, seq, dm = emb.shape
        np_ = patterns.size(0)
        vs = head_w.size(0)
        attract_every = ctx.attract_every
        beta = ctx.beta

        grad_logits = grad_logits.contiguous()

        d_h_init = torch.zeros(bs, dm, device=emb.device)
        d_emb = torch.zeros(bs, seq, dm, device=emb.device)
        d_patterns = torch.zeros(np_, dm, device=emb.device)
        d_wa = torch.zeros_like(wa)
        d_ba = torch.zeros_like(ba)
        d_wb = torch.zeros_like(wb)
        d_bb = torch.zeros_like(bb)
        d_wg = torch.zeros_like(wg)
        d_bg = torch.zeros_like(bg)
        d_wp = torch.zeros_like(wp)
        d_bp = torch.zeros_like(bp)
        d_wn = torch.zeros_like(wn)
        d_bn = torch.zeros_like(bn)
        d_head_w = torch.zeros_like(head_w)
        d_head_b = torch.zeros_like(head_b)
        d_sn_w = torch.zeros_like(sn_w)
        d_sn_b = torch.zeros_like(sn_b)
        d_slot_table = torch.zeros_like(slot_table)

        scratch_d_attn = torch.zeros(bs, np_, device=emb.device)
        scratch_d_scores = torch.zeros(bs, np_, device=emb.device)

        dll.launch_cann_sequence_backward(
            ctypes.c_void_p(grad_logits.data_ptr()),
            ctypes.c_void_p(saved_h_ssm.data_ptr()),
            ctypes.c_void_p(saved_gate_a.data_ptr()),
            ctypes.c_void_p(saved_gate_b.data_ptr()),
            ctypes.c_void_p(saved_alpha.data_ptr()),
            ctypes.c_void_p(saved_attn.data_ptr()),
            ctypes.c_void_p(saved_h_new.data_ptr()),
            ctypes.c_void_p(saved_cmean.data_ptr()),
            ctypes.c_void_p(saved_cinv_std.data_ptr()),
            ctypes.c_void_p(saved_hmean.data_ptr()),
            ctypes.c_void_p(saved_hinv_std.data_ptr()),
            ctypes.c_void_p(emb.data_ptr()),
            ctypes.c_void_p(x_tokens.data_ptr()),
            ctypes.c_void_p(patterns.data_ptr()),
            ctypes.c_void_p(slot_table.data_ptr()),
            ctypes.c_void_p(h_init.data_ptr()),
            ctypes.c_void_p(wa.data_ptr()), ctypes.c_void_p(ba.data_ptr()),
            ctypes.c_void_p(wb.data_ptr()), ctypes.c_void_p(bb.data_ptr()),
            ctypes.c_void_p(wg.data_ptr()), ctypes.c_void_p(bg.data_ptr()),
            ctypes.c_void_p(wp.data_ptr()), ctypes.c_void_p(bp.data_ptr()),
            ctypes.c_void_p(wn.data_ptr()), ctypes.c_void_p(bn.data_ptr()),
            ctypes.c_void_p(head_w.data_ptr()), ctypes.c_void_p(head_b.data_ptr()),
            ctypes.c_void_p(sn_w.data_ptr()), ctypes.c_void_p(sn_b.data_ptr()),
            ctypes.c_void_p(d_h_init.data_ptr()),
            ctypes.c_void_p(d_emb.data_ptr()),
            ctypes.c_void_p(d_patterns.data_ptr()),
            ctypes.c_void_p(d_wa.data_ptr()), ctypes.c_void_p(d_ba.data_ptr()),
            ctypes.c_void_p(d_wb.data_ptr()), ctypes.c_void_p(d_bb.data_ptr()),
            ctypes.c_void_p(d_wg.data_ptr()), ctypes.c_void_p(d_bg.data_ptr()),
            ctypes.c_void_p(d_wp.data_ptr()), ctypes.c_void_p(d_bp.data_ptr()),
            ctypes.c_void_p(d_wn.data_ptr()), ctypes.c_void_p(d_bn.data_ptr()),
            ctypes.c_void_p(d_head_w.data_ptr()), ctypes.c_void_p(d_head_b.data_ptr()),
            ctypes.c_void_p(d_sn_w.data_ptr()), ctypes.c_void_p(d_sn_b.data_ptr()),
            ctypes.c_void_p(d_slot_table.data_ptr()),
            ctypes.c_void_p(scratch_d_attn.data_ptr()),
            ctypes.c_void_p(scratch_d_scores.data_ptr()),
            bs, seq, dm, np_, vs,
            ctypes.c_float(beta),
            attract_every,
        )

        return (d_h_init, d_emb, None, d_patterns, d_slot_table,
                d_wa, d_ba, d_wb, d_bb, d_wg, d_bg, d_wp, d_bp, d_wn, d_bn,
                d_sn_w, d_sn_b, d_head_w, d_head_b,
                None, None)


class CANNStepCUDA(torch.autograd.Function):
    """Single CANN cell step.

    Forward: PyTorch (accurate, for training).
    CUDA kernel is used for full-sequence inference (use_cuda_seq=True),
    which bypasses autograd entirely.
    """
    @staticmethod
    def forward(ctx, h, x, patterns, wa, ba, wb, bb, wg, bg, wp, bp, wn, bn, beta):
        ctx.save_for_backward(h, x, patterns, wa, ba, wb, bb, wg, bg, wp, bp, wn, bn)
        ctx.beta_val = beta

        combined = torch.cat([h, x], dim=-1)
        a = torch.sigmoid(combined @ wa.T + ba)
        b = torch.sigmoid(combined @ wb.T + bb)
        h_ssm = a * h + b * (x @ wp.T + bp)
        scores = (h_ssm @ patterns.T) * beta
        an = torch.softmax(scores, dim=-1)
        al = torch.sigmoid(combined @ wg.T + bg)
        hn = h_ssm + al * (an @ patterns - h_ssm)
        mn = hn.mean(dim=-1, keepdim=True)
        return wn * (hn - mn) / torch.sqrt(hn.var(dim=-1, unbiased=False, keepdim=True) + 1e-5) + bn

    @staticmethod
    def backward(ctx, grad_output):
        h, x, patterns, wa, ba, wb, bb, wg, bg, wp, bp, wn, bn = ctx.saved_tensors
        beta = ctx.beta_val
        N = h.shape[-1]

        # Forward recomputation (no autograd, just compute values)
        combined = torch.cat([h, x], dim=-1)
        a = torch.sigmoid(combined @ wa.T + ba)
        b = torch.sigmoid(combined @ wb.T + bb)
        xp = x @ wp.T + bp
        h_ssm = a * h + b * xp
        scores = (h_ssm @ patterns.T) * beta
        attn = torch.softmax(scores, dim=-1)
        alpha = torch.sigmoid(combined @ wg.T + bg)
        hn = h_ssm + alpha * (attn @ patterns - h_ssm)
        inv_std = torch.rsqrt(hn.var(dim=-1, unbiased=False, keepdim=True) + 1e-5)
        mn = hn.mean(dim=-1, keepdim=True)
        h_norm = (hn - mn) * inv_std

        # LayerNorm backward
        d_hn = grad_output * wn
        m = d_hn.mean(dim=-1, keepdim=True)
        s = (d_hn * h_norm).mean(dim=-1, keepdim=True)
        d_h_new = (d_hn - m - h_norm * s) * inv_std

        # h_new = h_ssm + alpha * (attracted - h_ssm) — ELEMENT-WISE
        at = attn @ patterns
        d_alpha = d_h_new * (at - h_ssm)
        d_h_ssm = d_h_new * (1 - alpha)
        d_attracted = d_h_new * alpha

        # attracted = attn @ patterns
        d_attn = d_attracted @ patterns.T
        d_patterns = attn.reshape(attn.size(0), -1).t() @ d_attracted

        # Softmax backward: d_scores = attn * (d_attn - sum(attn * d_attn))
        d_scores = attn * (d_attn - (attn * d_attn).sum(dim=-1, keepdim=True))
        d_h_ssm = d_h_ssm + (d_scores * beta) @ patterns

        # h_ssm = a * h + b * xp
        d_h_grad = d_h_ssm * a
        d_xp = d_h_ssm * b
        d_x_grad = d_xp @ wp

        # Sigmoid backward for a, b, alpha (ELEMENT-WISE gates)
        d_pa = d_h_ssm * h * a * (1 - a)
        d_h_grad = d_h_grad + d_pa @ wa[:, :N]
        d_x_grad = d_x_grad + d_pa @ wa[:, N:]

        d_pb = d_h_ssm * xp * b * (1 - b)
        d_h_grad = d_h_grad + d_pb @ wb[:, :N]
        d_x_grad = d_x_grad + d_pb @ wb[:, N:]

        d_pal = d_alpha * alpha * (1 - alpha)
        d_h_grad = d_h_grad + d_pal @ wg[:, :N]
        d_x_grad = d_x_grad + d_pal @ wg[:, N:]

        return d_h_grad, d_x_grad, d_patterns, None, None, None, None, None, None, None, None, None, None, None


# ── JIT 细胞: SSM only ─────────────────────────────────────────
@torch.jit.script
def _cell_ssm(
    h: torch.Tensor, x: torch.Tensor,
    w_a: torch.Tensor, b_a: torch.Tensor,
    w_b: torch.Tensor, b_b: torch.Tensor,
    w_p: torch.Tensor, b_p: torch.Tensor,
    w_n: torch.Tensor, b_n: torch.Tensor,
):
    combined = torch.cat([h, x], dim=-1)
    a = torch.sigmoid(combined @ w_a.t() + b_a)
    b = torch.sigmoid(combined @ w_b.t() + b_b)
    h_ssm = a * h + b * (x @ w_p.t() + b_p)
    return torch.layer_norm(h_ssm, [h.shape[-1]], w_n, b_n, eps=1e-5)


# ── JIT 细胞: SSM + attractor ────────────────────────────────
@torch.jit.script
def _cell_full(
    h: torch.Tensor, x: torch.Tensor,
    patterns: torch.Tensor, beta: torch.Tensor,
    w_a: torch.Tensor, b_a: torch.Tensor,
    w_b: torch.Tensor, b_b: torch.Tensor,
    w_g: torch.Tensor, b_g: torch.Tensor,
    w_p: torch.Tensor, b_p: torch.Tensor,
    w_n: torch.Tensor, b_n: torch.Tensor,
):
    bsz = h.shape[0]
    combined = torch.cat([h, x], dim=-1)
    a = torch.sigmoid(combined @ w_a.t() + b_a)
    b = torch.sigmoid(combined @ w_b.t() + b_b)
    h_ssm = a * h + b * (x @ w_p.t() + b_p)

    pat = patterns.unsqueeze(0).expand(bsz, -1, -1)
    xi = h_ssm.unsqueeze(1)
    scores = xi @ pat.transpose(1, 2) * beta[0]
    attn = torch.softmax(scores, dim=-1)
    h_attracted = (attn @ pat).squeeze(1)

    alpha = torch.sigmoid(combined @ w_g.t() + b_g)
    h_new = h_ssm + alpha * (h_attracted - h_ssm)
    return torch.layer_norm(h_new, [h.shape[-1]], w_n, b_n, eps=1e-5)


def _cell_full_mimo(h, x, patterns_mimo, beta, w_a, b_a, w_b, b_b, w_g, b_g, w_p, b_p, w_n, b_n):
    """MIMO attractor: patterns_mimo [n_heads, np_per_head, dm] — parallel per-head retrieval."""
    n_heads, np_head, dm = patterns_mimo.shape
    bsz = h.shape[0]
    combined = torch.cat([h, x], dim=-1)
    a = torch.sigmoid(combined @ w_a.t() + b_a)
    b = torch.sigmoid(combined @ w_b.t() + b_b)
    h_ssm = a * h + b * (x @ w_p.t() + b_p)

    attracted_parts = []
    for i in range(n_heads):
        pat_i = patterns_mimo[i]
        scores_i = (h_ssm @ pat_i.t()) * beta[0]
        attn_i = torch.softmax(scores_i, dim=-1)
        attracted_parts.append(attn_i @ pat_i)

    attracted = torch.cat(attracted_parts, dim=-1)
    attracted = attracted.view(bsz, n_heads, dm).mean(dim=1)

    alpha = torch.sigmoid(combined @ w_g.t() + b_g)
    h_new = h_ssm + alpha * (attracted - h_ssm)
    return torch.layer_norm(h_new, [dm], w_n, b_n, eps=1e-5)


# ── JIT 全序列 forward ──────────────────────────────────────
@torch.jit.script
def _full_forward(
    x: torch.Tensor,
    embed_weight: torch.Tensor,
    slot_table: torch.Tensor,
    head_weight: torch.Tensor, head_bias: torch.Tensor,
    head_norm_w: torch.Tensor, head_norm_b: torch.Tensor,
    # cell weights (shared between ssm and full)
    patterns: torch.Tensor, beta: torch.Tensor,
    w_a: torch.Tensor, b_a: torch.Tensor,
    w_b: torch.Tensor, b_b: torch.Tensor,
    w_g: torch.Tensor, b_g: torch.Tensor,
    w_p: torch.Tensor, b_p: torch.Tensor,
    w_n: torch.Tensor, b_n: torch.Tensor,
    attract_every: int,
):
    bsz, seq_len = x.shape
    dm = head_norm_w.shape[0]
    emb = torch.nn.functional.embedding(x, embed_weight)
    h = torch.zeros(bsz, dm, device=x.device)
    logits = torch.zeros(bsz, seq_len, head_weight.shape[0], device=x.device)

    for t in range(seq_len - 1):
        if t % attract_every == (attract_every - 1):
            h = _cell_full(h, emb[:, t], patterns, beta,
                           w_a, b_a, w_b, b_b, w_g, b_g, w_p, b_p, w_n, b_n)
        else:
            h = _cell_ssm(h, emb[:, t],
                          w_a, b_a, w_b, b_b, w_p, b_p, w_n, b_n)
        hn = torch.layer_norm(h, [dm], head_norm_w, head_norm_b, eps=1e-5)
        logits[:, t] = hn @ head_weight.t() + head_bias

    # Last position: slot injection + full attractor
    i_ext = slot_table[x[:, -1]]
    h = _cell_full(h + i_ext, emb[:, -1], patterns, beta,
                   w_a, b_a, w_b, b_b, w_g, b_g, w_p, b_p, w_n, b_n)
    hn = torch.layer_norm(h, [dm], head_norm_w, head_norm_b, eps=1e-5)
    logits[:, -1] = hn @ head_weight.t() + head_bias

    return logits


# ── Python 版 (用于训练，因为 jit 无法处理 nn.Module 更新) ──
class CANNSSMCell(nn.Module):
    def __init__(self, d_model, n_patterns=4096, beta=1.0, attract_every=1, pattern_rank=0, n_heads=1):
        super().__init__()
        self.d_model = d_model
        self.attract_every = attract_every
        self.rank = pattern_rank
        self.n_heads = n_heads
        np_per_head = n_patterns // n_heads
        if self.rank > 0 and self.rank < n_patterns:
            self.U = nn.Parameter(torch.randn(n_patterns, self.rank) * 0.02)
            self.V = nn.Parameter(torch.randn(self.rank, d_model) * 0.02 / (self.rank ** 0.5))
            self.patterns = None
        elif n_heads > 1:
            # MIMO: separate patterns per head
            self.patterns = nn.Parameter(torch.randn(n_heads, np_per_head, d_model) * 0.02)
            self.U = None; self.V = None
        else:
            self.patterns = nn.Parameter(torch.randn(n_patterns, d_model) * 0.02)
            self.U = None; self.V = None
        self.register_buffer("beta_t", torch.tensor([beta]))
        self.gate_a = nn.Linear(d_model * 2, d_model)
        self.gate_b = nn.Linear(d_model * 2, d_model)
        self.gate_alpha = nn.Linear(d_model * 2, d_model)
        self.proj_in = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.use_cuda = _get_cuda_dll() is not None

    @property
    def effective_patterns(self):
        return self.patterns if self.patterns is not None else self.U @ self.V

    def forward(self, h, x, step=0):
        pat_eff = self.effective_patterns
        if step % self.attract_every == (self.attract_every - 1):
            if self.n_heads > 1 and self.patterns is not None:
                return _cell_full_mimo(h, x, self.patterns, self.beta_t,
                    self.gate_a.weight, self.gate_a.bias,
                    self.gate_b.weight, self.gate_b.bias,
                    self.gate_alpha.weight, self.gate_alpha.bias,
                    self.proj_in.weight, self.proj_in.bias,
                    self.norm.weight, self.norm.bias,
                )
            return _cell_full(
                h, x, pat_eff, self.beta_t,
                self.gate_a.weight, self.gate_a.bias,
                self.gate_b.weight, self.gate_b.bias,
                self.gate_alpha.weight, self.gate_alpha.bias,
                self.proj_in.weight, self.proj_in.bias,
                self.norm.weight, self.norm.bias,
            )
        return _cell_ssm(
            h, x,
            self.gate_a.weight, self.gate_a.bias,
            self.gate_b.weight, self.gate_b.bias,
            self.proj_in.weight, self.proj_in.bias,
            self.norm.weight, self.norm.bias,
        )


class RINASeqModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_patterns=4096,
                 beta=1.0, n_slots=4096, attract_every=1, pattern_rank=0, n_heads=1):
        super().__init__()
        self.d_model = d_model
        self.attract_every = attract_every
        self.embed = nn.Embedding(vocab_size, d_model)
        self.cell = CANNSSMCell(d_model, n_patterns=n_patterns,
                                 beta=beta, attract_every=attract_every,
                                 pattern_rank=pattern_rank, n_heads=n_heads)
        self.head = nn.Linear(d_model, vocab_size)
        self.state_norm = nn.LayerNorm(d_model)
        self.register_buffer("slot_table", torch.zeros(vocab_size, d_model))
        self.slot_proj = nn.Linear(d_model, d_model)

    def slot_write(self, key_id, value_id):
        with torch.no_grad():
            ve = self.embed(torch.tensor([value_id], device=self.slot_table.device))
            self.slot_table[key_id] = self.slot_proj(ve).squeeze(0)

    def forward(self, x, use_jit=False, use_cuda_seq=False):
        if not self.training:
            if use_cuda_seq and _get_cuda_dll() is not None:
                dll = _get_cuda_dll()
                dll.launch_cann_sequence.restype = None
                dll.launch_cann_sequence.argtypes = (
                    [ctypes.c_void_p] * 20 + [ctypes.c_int] * 5 + [ctypes.c_float] + [ctypes.c_int]
                )
                bs, seq = x.shape
                dm = self.d_model
                np_ = self.cell.patterns.size(0)
                vs = self.head.weight.size(0)
                h_init = torch.zeros(bs, dm, device=x.device)
                emb = self.embed(x)
                logits = torch.zeros(bs, seq, vs, device=x.device)
                dll.launch_cann_sequence(
                    ctypes.c_void_p(h_init.data_ptr()),
                    ctypes.c_void_p(emb.data_ptr()),
                    ctypes.c_void_p(x.to(torch.int32).data_ptr()),  # token IDs
                    ctypes.c_void_p(self.cell.patterns.data_ptr()),
                    ctypes.c_void_p(self.slot_table.data_ptr()),
                    ctypes.c_void_p(self.cell.gate_a.weight.data_ptr()),
                    ctypes.c_void_p(self.cell.gate_a.bias.data_ptr()),
                    ctypes.c_void_p(self.cell.gate_b.weight.data_ptr()),
                    ctypes.c_void_p(self.cell.gate_b.bias.data_ptr()),
                    ctypes.c_void_p(self.cell.gate_alpha.weight.data_ptr()),
                    ctypes.c_void_p(self.cell.gate_alpha.bias.data_ptr()),
                    ctypes.c_void_p(self.cell.proj_in.weight.data_ptr()),
                    ctypes.c_void_p(self.cell.proj_in.bias.data_ptr()),
                    ctypes.c_void_p(self.cell.norm.weight.data_ptr()),
                    ctypes.c_void_p(self.cell.norm.bias.data_ptr()),
                    ctypes.c_void_p(self.head.weight.data_ptr()),
                    ctypes.c_void_p(self.head.bias.data_ptr()),
                    ctypes.c_void_p(self.state_norm.weight.data_ptr()),
                    ctypes.c_void_p(self.state_norm.bias.data_ptr()),
                    ctypes.c_void_p(logits.data_ptr()),
                    bs, seq, dm, np_, vs,
                    ctypes.c_float(self.cell.beta_t[0].item()),
                    self.attract_every,
                )
                return logits

            if use_jit:
                return _full_forward(
                    x, self.embed.weight, self.slot_table,
                    self.head.weight, self.head.bias,
                    self.state_norm.weight, self.state_norm.bias,
                    self.cell.effective_patterns, self.cell.beta_t,
                    self.cell.gate_a.weight, self.cell.gate_a.bias,
                    self.cell.gate_b.weight, self.cell.gate_b.bias,
                    self.cell.gate_alpha.weight, self.cell.gate_alpha.bias,
                    self.cell.proj_in.weight, self.cell.proj_in.bias,
                    self.cell.norm.weight, self.cell.norm.bias,
                    self.attract_every,
                )

        # Training: use CUDA sequence v2 if available, else Python loop
        if self.training and _setup_cuda_seq_v2():
            bsz, seq_len = x.shape
            dm = self.d_model
            h_init = torch.zeros(bsz, dm, device=x.device)
            emb = self.embed(x)
            xt = x.to(torch.int32)
            beta = self.cell.beta_t[0].item()

            logits = CANNSequenceCUDA.apply(
                h_init, emb, xt, 
                self.cell.patterns, self.slot_table,
                self.cell.gate_a.weight, self.cell.gate_a.bias,
                self.cell.gate_b.weight, self.cell.gate_b.bias,
                self.cell.gate_alpha.weight, self.cell.gate_alpha.bias,
                self.cell.proj_in.weight, self.cell.proj_in.bias,
                self.cell.norm.weight, self.cell.norm.bias,
                self.state_norm.weight, self.state_norm.bias,
                self.head.weight, self.head.bias,
                beta, self.attract_every,
            )
            return logits

        bsz, seq_len = x.shape
        emb = self.embed(x)
        h = torch.zeros(bsz, self.d_model, device=x.device)
        logits = []

        for t in range(seq_len - 1):
            h = self.cell(h, emb[:, t, :], step=t)
            logits.append(self.head(self.state_norm(h)))

        i_ext = self.slot_table[x[:, -1]]
        h = self.cell(h + i_ext, emb[:, -1, :], step=seq_len - 1)
        logits.append(self.head(self.state_norm(h)))

        return torch.stack(logits, dim=1)

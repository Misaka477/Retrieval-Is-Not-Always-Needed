"""
CUDA Graph for RINA skip path (gate+norm+head, no branches).
Captures the 74% of steps where attractor is skipped.
"""
import torch


def make_skip_graph(model, device="cuda", batch_size=8, d_model=840, vocab_size=4096):
    """Capture skip path as CUDA graph: gate(h, x_emb) -> norm -> head."""
    h = torch.zeros(batch_size, d_model, device=device)
    x_emb = torch.randn(batch_size, d_model, device=device)

    graph = torch.cuda.CUDAGraph()
    static_h = h.clone()
    static_x_emb = x_emb.clone()
    static_logit = torch.zeros(batch_size, vocab_size, device=device)

    cell = model.cell
    with torch.cuda.graph(graph):
        combined = torch.cat([static_h, static_x_emb], dim=-1)
        a = torch.sigmoid(cell.gate_a(combined))
        b = torch.sigmoid(cell.gate_b(combined))
        xp = cell.proj_in(static_x_emb)
        h_new = a * static_h + b * xp
        h_new = cell.norm(h_new)
        logit = model.head(model.state_norm(h_new))
        static_logit.copy_(logit)

    def replay(h_in, x_emb_in):
        static_h.copy_(h_in)
        static_x_emb.copy_(x_emb_in)
        graph.replay()
        h_out = cell.norm(cell.gate_a(torch.cat([static_h, static_x_emb], -1)) * static_h + cell.gate_b(torch.cat([static_h, static_x_emb], -1)) * cell.proj_in(static_x_emb))
        return h_out, static_logit.clone()

    return replay

"""
Basin drift tracker — measure pattern stability during training.
Pattern drift = ||P_epoch - P_previous|| / ||P_previous||
Converged when drift < threshold (e.g. 1%).

Also tracks basin norm distribution and coverage analysis
for plateau diagnosis.
"""
import torch


def compute_drift(P_prev, P_curr):
    P_prev_n = P_prev / P_prev.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    P_curr_n = P_curr / P_curr.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    avg_dot = (P_prev_n * P_curr_n).sum(dim=-1).mean().item()
    frob_drift = (P_curr - P_prev).norm().item() / P_prev.norm().item()
    return {"avg_cos": avg_dot, "frob_drift": frob_drift}


def compute_norm_stats(P):
    norms = P.norm(dim=-1)
    return {
        "norms_mean": norms.mean().item(),
        "norms_std": norms.std().item(),
        "norms_min": norms.min().item(),
        "norms_max": norms.max().item(),
        "dead_frac": (norms < 0.01).float().mean().item(),
        "runaway_frac": (norms > norms.mean() + 5 * norms.std()).float().mean().item(),
    }


def compute_coverage(P):
    norms = P.norm(dim=-1).clamp(min=1e-8)
    pairwise = (P @ P.T) / (norms.unsqueeze(-1) * norms.unsqueeze(0))
    triu = pairwise[torch.triu(torch.ones_like(pairwise), diagonal=1) == 1]
    self_sim_mean = triu.mean().item() if triu.numel() > 0 else 0.0
    self_sim_max = triu.max().item() if triu.numel() > 0 else 0.0
    U, S, Vt = torch.linalg.svd(P, full_matrices=False); S[S < 1e-10] = 1e-10
    cumvar = (S.cumsum(0) / S.sum()).tolist()
    eff_rank_95 = next((i for i, v in enumerate(cumvar) if v >= 0.95), len(S))
    return {
        "self_sim_mean": self_sim_mean,
        "self_sim_max": self_sim_max,
        "eff_rank_95": eff_rank_95,
        "condition_number": (S[0] / S[-1]).item(),
    }


class DriftTracker:
    def __init__(self, compute_coverage=True):
        self.P_prev = None
        self.history = []
        self._compute_cov = compute_coverage

    def step(self, patterns):
        P = patterns.detach().cpu()
        cov_stats = compute_coverage(P) if self._compute_cov else {}
        norm_stats = compute_norm_stats(P)
        if self.P_prev is not None:
            drift = compute_drift(self.P_prev, P)
            metrics = {**drift, **norm_stats, **cov_stats}
            self.history.append(metrics)
        else:
            metrics = {"avg_cos": 1.0, "frob_drift": 0.0, **norm_stats, **cov_stats}
        self.P_prev = P.clone()
        return metrics

    def summary(self):
        if not self.history:
            return "No drift data"
        f = self.history[-1]
        out = (f"basins: {len(self.history)} epochs | "
               f"drift[cos={f['avg_cos']:.4f}, frob={f['frob_drift']:.4f}] | "
               f"norms[mu={f['norms_mean']:.3f}, sig={f['norms_std']:.3f}, "
               f"dead={f['dead_frac']*100:.1f}%]")
        if "eff_rank_95" in f:
            out += (f" | cov[mu_cos={f['self_sim_mean']:.3f}, "
                    f"r95={f['eff_rank_95']}, "
                    f"kappa={f['condition_number']:.0f}]")
        return out

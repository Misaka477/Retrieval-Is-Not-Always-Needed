"""自适应采样: Adaptive Temperature + Top-P + multinomial."""
import math, torch

def sample(logits, temp=0.8, top_p=0.9, temp_min=0.3, temp_max=1.5):
    logits = logits.float()
    p = torch.softmax(logits / temp, dim=-1)
    H = -(p * torch.log(p.clamp(min=1e-10))).sum(-1).mean().item()
    H_target = 0.8 * math.log(logits.shape[-1])
    temp_dyn = temp + 0.3 * (H_target - H)
    temp_dyn = max(temp_min, min(temp_max, temp_dyn))
    p = torch.softmax(logits / temp_dyn, dim=-1)
    sorted_p, _ = p.sort(descending=True)
    cumsum = sorted_p.cumsum(dim=-1)
    mask = cumsum - sorted_p > top_p
    p[mask] = 0.0
    p /= p.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    return torch.multinomial(p, 1)

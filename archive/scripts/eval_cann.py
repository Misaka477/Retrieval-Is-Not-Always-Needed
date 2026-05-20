"""Test CANN-SSM trained on TinyStories."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tokenizers import ByteLevelBPETokenizer
import torch
from modules.cann_ssm import RINASeqModel

device = "cuda"
tok = ByteLevelBPETokenizer.from_file(
    "D:\\Software_Development\\Project\\RINA_Project\\checkpoints\\ts-vocab.json",
    "D:\\Software_Development\\Project\\RINA_Project\\checkpoints\\ts-merges.txt")
V = tok.get_vocab_size()
print(f"Tokenizer: vocab={V}")

m = RINASeqModel(V, d_model=256, n_patterns=4096, beta=0.5, attract_every=2, n_slots=0).to(device)
m.load_state_dict(torch.load("D:\\Software_Development\\Project\\RINA_Project\\checkpoints\\cann_final.pt",
                              map_location=device, weights_only=True))
m.eval()

for p in ["Once upon a time", "One day", "The little girl", "A big dog"]:
    ids = tok.encode(p).ids
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        for _ in range(60):
            nid = m(x)[0, -1].argmax().item()
            x = torch.cat([x, torch.tensor([[nid]], device=device)], dim=1)
    print(f"\nPrompt: {p}")
    print(f"Gen:    {tok.decode(x[0].tolist())[:200]}")

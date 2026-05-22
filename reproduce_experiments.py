"""
RINA reproducible experiments — run all key results in order.
Usage: python reproduce_experiments.py

Prerequisites:
  1. pip install -r requirements.txt
  2. Download checkpoints from GitHub Releases into checkpoints/
     (or the scripts will print download instructions)
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def run(cmd, desc):
    print(f"\n  >> {desc}")
    print(f"  Running: {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        print(f"  [WARN] exited with code {rc}")
    return rc

# ── 1. Attention-based slot memory (no checkpoint needed) ──
section("1. Attention-based slot memory (原理验证, ~30s)")
print("  This experiment is self-contained — no checkpoint download needed.")
run("python experiments/attn_slot.py", "Attention slot retrieval noise test")

# ── 2. WikiText validation ppl ──
section("2. WikiText-103 validation ppl (~5min)")
if os.path.exists("checkpoints/cann_snn15m_v2_slot_ep13.pt"):
    run("python scripts/bench_wikitext_valid.py", "RINA vs GPT-2 vs LLaMA on WikiText")
else:
    print("  [SKIP] checkpoint not found. Download from GitHub Releases:")
    print("  https://github.com/Misaka477/Retrieval-Is-Not-Always-Needed/releases")
    print("  Required: cann_snn15m_v2_slot_ep13.pt")

# ── 3. Code zero-shot ppl ──
section("3. Code zero-shot ppl (~5min)")
if os.path.exists("checkpoints/code_seq256_resume.pt"):
    run("python scripts/bench_code_ppl.py --seq 64 --th 1.0", "Code perplexity (zero-shot)")
else:
    print("  [SKIP] code-seq256 checkpoint not found. Download from GitHub Releases.")
    print("  Required: code_seq256_resume.pt")

# ── 4. Mamba-130M baseline (~5min, downloads model) ──
section("4. Mamba-130M baseline (~5min, downloads ~500MB)")
print("  Downloads Mamba-130M from HuggingFace Hub.")
run("python scripts/bench_mamba_130m.py", "Mamba vs RINA comparison")

# ── 5. Force attractor experiment ──
section("5. Attractor contribution (~2min)")
if os.path.exists("checkpoints/cann_snn15m_v2_slot_ep13.pt"):
    run("python experiments/force_attractor.py", "Forced attractor vs normal")
else:
    print("  [SKIP] checkpoint not found.")

# ── Summary ──
section("Summary")
print("""
Key results to compare:

  RINA 15M (4K vocab)          GPT-2 124M (50K)       LLaMA 1B (128K)       Mamba-130M (50K)
  ──────────────────────────────────────────────────────────────────────────────────────────────
  WikiText-103: 34.6           25.4                    11.4                   19.1
  Code zero-shot: 65.8         14432                   19.7                   —
  Slot retrieval noise 2.0: 5/5  —                       —                     —
  Attractor contribution: 0 ppl —                       —                     —

  See docs/RINA实验日志.md for full experiment log.
  See docs/RINA_实验总览.md for structured experiment overview.
""")
print("Done.")

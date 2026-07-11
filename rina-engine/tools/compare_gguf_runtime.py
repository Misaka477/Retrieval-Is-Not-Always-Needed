#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd, env):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode, proc.stdout


def parse_llama_ppl(output):
    match = re.search(r"Final estimate: PPL = ([0-9.eE+-]+)(?: \+/- ([0-9.eE+-]+))?", output)
    if match:
        return {
            "ppl": float(match.group(1)),
            "ppl_std": float(match.group(2)) if match.group(2) else None,
        }
    chunk_matches = re.findall(r"\[[0-9]+\]([0-9.eE+-]+)", output)
    if chunk_matches:
        return {
            "ppl": float(chunk_matches[-1]),
            "ppl_std": None,
        }
    return None


def parse_rina_ppl(output):
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ppl" in data:
                return data
    return None


def parse_prefixed_json(output, prefix):
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix):].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    return None


def parse_generated_text_smoke(output):
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith((
            "Loading GGUF model",
            "llama runtime",
            "gguf prompt text",
            "gguf prompt=",
            "tokens:",
            "runtime prefill",
            "logits[",
            "RINA_INFER_PERF",
        )):
            continue
        return True
    return False


def parse_json_array(output):
    start = output.find("[")
    end = output.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(output[start:end + 1])
    except json.JSONDecodeError:
        return None


def parse_llama_bench(output):
    data = parse_json_array(output)
    if not isinstance(data, list):
        return None
    result = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        n_prompt = int(row.get("n_prompt") or 0)
        n_gen = int(row.get("n_gen") or 0)
        avg_ts = row.get("avg_ts")
        if avg_ts is None:
            continue
        if n_prompt > 0 and n_gen == 0:
            result["prompt_tokens_per_second"] = float(avg_ts)
            result["prompt_tokens"] = n_prompt
        elif n_gen > 0:
            result["decode_tokens_per_second"] = float(avg_ts)
            result["decode_tokens"] = n_gen
    return result or None


def parse_llama_perf(output):
    prompt = re.search(r"prompt eval time =\s*([0-9.eE+-]+) ms /\s*([0-9]+) tokens .*?([0-9.eE+-]+) tokens per second", output)
    decode = re.search(r"eval time =\s*([0-9.eE+-]+) ms /\s*([0-9]+) runs .*?([0-9.eE+-]+) tokens per second", output)
    if not prompt and not decode:
        return None
    return {
        "prompt_eval_ms": float(prompt.group(1)) if prompt else None,
        "prompt_tokens": int(prompt.group(2)) if prompt else None,
        "prompt_tokens_per_second": float(prompt.group(3)) if prompt else None,
        "decode_ms": float(decode.group(1)) if decode else None,
        "decode_runs": int(decode.group(2)) if decode else None,
        "decode_tokens_per_second": float(decode.group(3)) if decode else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare RINA GGUF runtime perplexity against llama.cpp.")
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--text", default="rina-engine/eval/data/wikitext-2-raw/wiki.test.raw", help="Evaluation text path")
    parser.add_argument("--context", type=int, default=512, help="Context size")
    parser.add_argument("--stride", type=int, default=0, help="Optional llama.cpp ppl stride")
    parser.add_argument("--llama-dir", default="external/llama.cpp", help="llama.cpp directory relative to repo root")
    parser.add_argument("--rina-dir", default="rina-engine", help="RINA engine directory relative to repo root")
    parser.add_argument("--cuda-bin", default="/home/aquama/miniconda3/envs/natalia/bin", help="CUDA env bin path")
    parser.add_argument("--ngl", default="all", help="llama.cpp GPU layer offload setting")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Allowed relative PPL error")
    parser.add_argument("--infer-steps", type=int, default=0, help="Also run RINA GGUF generation benchmark")
    parser.add_argument("--infer-prompt", default="The capital of France is", help="Text prompt for RINA generation benchmark")
    parser.add_argument("--infer-ids", default=None, help="Optional token ids for low-level RINA ids API smoke test")
    parser.add_argument("--bench-prompt", type=int, default=0, help="Also run RINA prompt eval benchmark")
    parser.add_argument("--bench-reps", type=int, default=3, help="RINA prompt benchmark repetitions")
    parser.add_argument("--show-output", action="store_true", help="Print raw command output")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    llama_dir = (repo / args.llama_dir).resolve()
    rina_dir = (repo / args.rina_dir).resolve()
    model = str(Path(args.model).resolve())
    text_path = Path(args.text)
    if not text_path.is_absolute():
        text_path = repo / text_path
    text = str(text_path.resolve())

    env = os.environ.copy()
    env["PATH"] = f"{args.cuda_bin}:{env.get('PATH', '')}"

    llama_cmd = [
        str(llama_dir / "build_cuda/bin/llama-perplexity"),
        "-m", model,
        "-f", text,
        "-c", str(args.context),
        "-b", str(args.context),
        "-ngl", args.ngl,
        "-fa", "on",
    ]
    if args.stride > 0:
        llama_cmd.extend(["--ppl-stride", str(args.stride)])

    llama_bench_cmd = [
        str(llama_dir / "build_cuda/bin/llama-bench"),
        "-m", model,
        "-p", str(args.bench_prompt if args.bench_prompt > 0 else 0),
        "-n", str(args.infer_steps if args.infer_steps > 0 else 0),
        "-r", str(max(1, args.bench_reps)),
        "-b", str(max(512, args.bench_prompt if args.bench_prompt > 0 else args.context)),
        "-ub", "512",
        "-ngl", "-1",
        "-fa", "on",
        "-sm", "none",
        "-o", "json",
    ]

    rina_cmd = [
        str(rina_dir / "build/eval_ppl"),
        "--gguf",
        "--model", model,
        "--text", text,
        "--context", str(args.context),
    ]
    if args.stride > 0:
        rina_cmd.extend(["--stride", str(args.stride)])

    infer_cmd = [
        str(rina_dir / "build/rina_infer"),
        "--gguf",
        "--model", model,
    ]
    if args.infer_ids:
        infer_cmd.extend(["--ids", args.infer_ids])
    else:
        infer_cmd.extend(["--prompt", args.infer_prompt])
    infer_cmd.extend(["--steps", str(args.infer_steps)])
    prompt_bench_cmd = [
        str(rina_dir / "build/rina_infer"),
        "--gguf",
        "--model", model,
    ]
    if args.infer_ids:
        prompt_bench_cmd.extend(["--ids", args.infer_ids])
    else:
        prompt_bench_cmd.extend(["--prompt", args.infer_prompt])
    prompt_bench_cmd.extend([
        "--steps", "0",
        "--bench-prompt", str(args.bench_prompt),
        "--bench-reps", str(args.bench_reps),
    ])

    llama_rc, llama_out = run(llama_cmd, llama_dir, env)
    rina_rc, rina_out = run(rina_cmd, rina_dir, env)
    llama_bench_rc, llama_bench_out = (0, "")
    if args.infer_steps > 0 or args.bench_prompt > 0:
        llama_bench_rc, llama_bench_out = run(llama_bench_cmd, llama_dir, env)
    infer_rc, infer_out = (0, "")
    if args.infer_steps > 0:
        infer_rc, infer_out = run(infer_cmd, rina_dir, env)
    prompt_bench_rc, prompt_bench_out = (0, "")
    if args.bench_prompt > 0:
        prompt_bench_rc, prompt_bench_out = run(prompt_bench_cmd, rina_dir, env)

    if args.show_output:
        print("=== llama.cpp ===", file=sys.stderr)
        print(llama_out, file=sys.stderr)
        print("=== RINA ===", file=sys.stderr)
        print(rina_out, file=sys.stderr)
        if args.infer_steps > 0:
            print("=== RINA infer ===", file=sys.stderr)
            print(infer_out, file=sys.stderr)
        if args.bench_prompt > 0:
            print("=== RINA prompt bench ===", file=sys.stderr)
            print(prompt_bench_out, file=sys.stderr)
        if args.infer_steps > 0 or args.bench_prompt > 0:
            print("=== llama-bench ===", file=sys.stderr)
            print(llama_bench_out, file=sys.stderr)

    llama = parse_llama_ppl(llama_out)
    rina = parse_rina_ppl(rina_out)
    llama_perf = parse_llama_perf(llama_out)
    llama_bench = parse_llama_bench(llama_bench_out) if (args.infer_steps > 0 or args.bench_prompt > 0) else None
    rina_perf = parse_prefixed_json(rina_out, "RINA_PPL_PERF")
    rina_infer_perf = parse_prefixed_json(infer_out, "RINA_INFER_PERF") if args.infer_steps > 0 else None
    rina_prompt_bench = parse_prefixed_json(prompt_bench_out, "RINA_PROMPT_BENCH") if args.bench_prompt > 0 else None
    if llama_rc != 0 or not llama:
        print(json.dumps({"ok": False, "error": "llama.cpp PPL failed", "returncode": llama_rc}, indent=2))
        return 1
    if rina_rc != 0 or not rina:
        print(json.dumps({"ok": False, "error": "RINA PPL failed", "returncode": rina_rc}, indent=2))
        return 1
    if args.infer_steps > 0 and (infer_rc != 0 or not rina_infer_perf):
        print(json.dumps({"ok": False, "error": "RINA infer benchmark failed", "returncode": infer_rc}, indent=2))
        return 1
    detokenize_smoke = parse_generated_text_smoke(infer_out) if args.infer_steps > 0 else None
    if args.infer_steps > 0 and not detokenize_smoke:
        print(json.dumps({"ok": False, "error": "RINA detokenize smoke failed", "returncode": infer_rc}, indent=2))
        return 1
    if args.bench_prompt > 0 and (prompt_bench_rc != 0 or not rina_prompt_bench):
        print(json.dumps({"ok": False, "error": "RINA prompt benchmark failed", "returncode": prompt_bench_rc}, indent=2))
        return 1
    if (args.infer_steps > 0 or args.bench_prompt > 0) and (llama_bench_rc != 0 or not llama_bench):
        print(json.dumps({"ok": False, "error": "llama-bench failed", "returncode": llama_bench_rc}, indent=2))
        return 1

    rel_error = abs(rina["ppl"] - llama["ppl"]) / llama["ppl"]
    prompt_ratio = None
    if llama_bench and rina_prompt_bench and llama_bench.get("prompt_tokens_per_second"):
        prompt_ratio = rina_prompt_bench["tokens_per_second"] / llama_bench["prompt_tokens_per_second"]
    decode_ratio = None
    if llama_bench and rina_infer_perf and llama_bench.get("decode_tokens_per_second"):
        decode_ratio = rina_infer_perf["decode_tokens_per_second"] / llama_bench["decode_tokens_per_second"]
    result = {
        "ok": rel_error <= args.tolerance,
        "context": args.context,
        "stride": args.stride,
        "llama": llama,
        "llama_bench": llama_bench,
        "llama_perf": llama_perf,
        "performance_ratio": {
            "prompt": prompt_ratio,
            "decode": decode_ratio,
        },
        "rina": rina,
        "rina_infer_perf": rina_infer_perf,
        "rina_detokenize_smoke": detokenize_smoke,
        "rina_perf": rina_perf,
        "rina_prompt_bench": rina_prompt_bench,
        "relative_error": rel_error,
        "tolerance": args.tolerance,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

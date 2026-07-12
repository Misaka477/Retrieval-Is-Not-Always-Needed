#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


SMOKE_LINE = (
    "The capital of France is Paris. The Eiffel Tower is a landmark in the city. "
    "This short text is used for a GGUF runtime smoke test. "
    "The model should assign reasonable probabilities to ordinary English text.\n"
)

TOKEN_EVENT_KEYS = {"event", "index", "id", "text", "logprob"}
PERF_EVENT_KEYS = {
    "event",
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "decode_tokens",
}


def run(cmd, cwd, env, timeout, label):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    return {
        "label": label,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def parse_prefixed_json(output, prefix):
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix):].strip()
        return json.loads(payload)
    return None


def parse_json_stdout_lines(output):
    rows = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def command_text(cmd):
    return " ".join(str(x) for x in cmd)


def check_command_ok(result):
    require(
        result["returncode"] == 0,
        f"{result['label']} failed with rc={result['returncode']}\n"
        f"cmd: {command_text(result['cmd'])}\n"
        f"stdout:\n{result['stdout']}\n"
        f"stderr:\n{result['stderr']}",
    )


def write_smoke_text(path, repeats):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SMOKE_LINE * repeats, encoding="utf-8")


def test_default_text(rina_infer, model, prompt, steps, ctx, cwd, env, timeout):
    result = run(
        [
            str(rina_infer),
            "--gguf",
            "--model", str(model),
            "--prompt", prompt,
            "--steps", str(steps),
            "--ctx", str(ctx),
        ],
        cwd,
        env,
        timeout,
        "default_text",
    )
    check_command_ok(result)
    stdout_lines = [line.strip() for line in result["stdout"].splitlines() if line.strip()]
    require(stdout_lines, "default text output is empty")
    require(not all(line.startswith("tokens:") for line in stdout_lines), "default output is still token-id only")
    perf = parse_prefixed_json(result["stderr"], "RINA_INFER_PERF")
    require(perf is not None, "missing RINA_INFER_PERF in default text run")
    require(perf.get("decode_tokens") == steps, f"decode_tokens mismatch: {perf.get('decode_tokens')} != {steps}")
    return {
        "stdout": result["stdout"].strip(),
        "perf": perf,
    }


def test_output_modes(rina_infer, model, prompt, steps, ctx, cwd, env, timeout):
    outputs = {}
    for flag in ("--stream", "--no-stream"):
        result = run(
            [
                str(rina_infer), "--gguf", "--model", str(model),
                "--prompt", prompt, "--steps", str(steps), "--ctx", str(ctx),
                flag, "--quiet",
            ],
            cwd, env, timeout, flag[2:],
        )
        check_command_ok(result)
        require(result["stdout"].strip(), f"{flag} text output is empty")
        require(result["stdout"].encode("utf-8").decode("utf-8") == result["stdout"], f"{flag} output is not valid UTF-8")
        stderr_lines = [line for line in result["stderr"].splitlines() if line.strip()]
        require(len(stderr_lines) == 1 and stderr_lines[0].startswith("RINA_INFER_PERF "), f"{flag} --quiet emitted diagnostics: {result['stderr']}")
        outputs[flag[2:]] = result["stdout"].strip()
    require(outputs["stream"] == outputs["no-stream"], "stream and no-stream generated text differ")
    return outputs


def test_token_ids(rina_infer, model, prompt, steps, ctx, cwd, env, timeout):
    result = run(
        [
            str(rina_infer),
            "--gguf",
            "--model", str(model),
            "--prompt", prompt,
            "--steps", str(steps),
            "--ctx", str(ctx),
            "--print-token-ids",
        ],
        cwd,
        env,
        timeout,
        "print_token_ids",
    )
    check_command_ok(result)
    stdout_lines = [line.strip() for line in result["stdout"].splitlines() if line.strip()]
    token_lines = [line for line in stdout_lines if line.startswith("tokens:")]
    text_lines = [line for line in stdout_lines if not line.startswith("tokens:")]
    require(text_lines, "--print-token-ids did not preserve generated text output")
    require(len(token_lines) == 1, "--print-token-ids did not emit exactly one token line")
    token_ids = [int(x) for x in re.findall(r"-?\d+", token_lines[0])]
    require(len(token_ids) == steps, f"token id count mismatch: {len(token_ids)} != {steps}")
    perf = parse_prefixed_json(result["stderr"], "RINA_INFER_PERF")
    require(perf is not None, "missing RINA_INFER_PERF in token id run")
    return {
        "text": "\n".join(text_lines),
        "tokens": token_ids,
        "perf": perf,
    }


def test_jsonl(rina_infer, model, prompt, steps, ctx, cwd, env, timeout):
    result = run(
        [
            str(rina_infer),
            "--gguf",
            "--model", str(model),
            "--prompt", prompt,
            "--steps", str(steps),
            "--ctx", str(ctx),
            "--jsonl",
        ],
        cwd,
        env,
        timeout,
        "jsonl",
    )
    check_command_ok(result)
    rows = parse_json_stdout_lines(result["stdout"])
    token_events = [row for row in rows if row.get("event") == "token"]
    perf_events = [row for row in rows if row.get("event") == "perf"]
    require(len(token_events) == steps, f"JSONL token event count mismatch: {len(token_events)} != {steps}")
    require(len(perf_events) == 1, f"JSONL perf event count mismatch: {len(perf_events)} != 1")
    for i, row in enumerate(token_events):
        require(set(row) == TOKEN_EVENT_KEYS, f"JSONL token schema mismatch at {i}: {sorted(row)}")
        require(row.get("index") == i, f"JSONL token index mismatch at {i}: {row.get('index')}")
        require(isinstance(row.get("id"), int), f"JSONL token id is not int at {i}")
        require(isinstance(row.get("text"), str), f"JSONL token text is not string at {i}")
        require(row.get("logprob") is None or isinstance(row.get("logprob"), (int, float)), f"JSONL token logprob has invalid type at {i}")
    perf_event = perf_events[0]
    require(set(perf_event) == PERF_EVENT_KEYS, f"JSONL perf schema mismatch: {sorted(perf_event)}")
    require(perf_event.get("decode_tokens") == steps, "JSONL perf decode_tokens mismatch")
    require(isinstance(perf_event.get("prefill_tokens_per_second"), (int, float)), "JSONL prefill rate is not numeric")
    require(isinstance(perf_event.get("decode_tokens_per_second"), (int, float)), "JSONL decode rate is not numeric")
    perf = parse_prefixed_json(result["stderr"], "RINA_INFER_PERF")
    require(perf is not None, "missing RINA_INFER_PERF in JSONL run")
    return {
        "events": rows,
        "perf": perf,
    }


def test_sampler_determinism(rina_infer, model, prompt, steps, ctx, cwd, env, timeout):
    cmd = [
        str(rina_infer), "--gguf", "--model", str(model),
        "--prompt", prompt, "--steps", str(steps), "--ctx", str(ctx),
        "--temp", "0.8", "--top-k", "40", "--top-p", "0.9",
        "--seed", "42", "--no-stream", "--quiet", "--print-token-ids",
    ]
    first = run(cmd, cwd, env, timeout, "sampler_seed_first")
    second = run(cmd, cwd, env, timeout, "sampler_seed_second")
    check_command_ok(first)
    check_command_ok(second)
    require(first["stdout"] == second["stdout"], "seeded sampler output is not deterministic")
    perf = parse_prefixed_json(first["stderr"], "RINA_INFER_PERF")
    require(perf is not None, "missing RINA_INFER_PERF in sampler determinism run")
    require(perf.get("decode_tokens") == steps, "sampler determinism decode_tokens mismatch")
    return {
        "stdout": first["stdout"].strip(),
        "perf": perf,
    }


def test_sampler_params(rina_infer, model, prompt, steps, ctx, cwd, env, timeout):
    result = run(
        [
            str(rina_infer), "--gguf", "--model", str(model),
            "--prompt", prompt, "--steps", str(steps), "--ctx", str(ctx),
            "--temp", "0.8", "--top-k", "40", "--top-p", "0.9",
            "--min-p", "0.05", "--typical-p", "0.95",
            "--repeat-penalty", "1.05", "--frequency-penalty", "0.10",
            "--presence-penalty", "0.10", "--repeat-last-n", "64",
            "--seed", "7", "--no-stream", "--quiet",
        ],
        cwd, env, timeout, "sampler_params",
    )
    check_command_ok(result)
    require(result["stdout"].strip(), "sampler params output is empty")
    perf = parse_prefixed_json(result["stderr"], "RINA_INFER_PERF")
    require(perf is not None, "missing RINA_INFER_PERF in sampler params run")
    require(perf.get("decode_tokens") == steps, "sampler params decode_tokens mismatch")
    return {
        "stdout": result["stdout"].strip(),
        "perf": perf,
    }


def test_stop_strings(rina_infer, model, prompt, steps, ctx, cwd, env, timeout):
    text_result = run(
        [
            str(rina_infer), "--gguf", "--model", str(model),
            "--prompt", prompt, "--steps", str(steps), "--ctx", str(ctx),
            "--stop", ".", "--quiet",
        ],
        cwd, env, timeout, "stop_text",
    )
    check_command_ok(text_result)
    require("." not in text_result["stdout"], f"stop string leaked into text output: {text_result['stdout']}")
    text_perf = parse_prefixed_json(text_result["stderr"], "RINA_INFER_PERF")
    require(text_perf is not None, "missing RINA_INFER_PERF in stop text run")
    require(0 < text_perf.get("decode_tokens", 0) < steps, "stop text run did not stop early")

    jsonl_result = run(
        [
            str(rina_infer), "--gguf", "--model", str(model),
            "--prompt", prompt, "--steps", str(steps), "--ctx", str(ctx),
            "--stop", ".", "--jsonl", "--quiet",
        ],
        cwd, env, timeout, "stop_jsonl",
    )
    check_command_ok(jsonl_result)
    rows = parse_json_stdout_lines(jsonl_result["stdout"])
    token_text = "".join(row.get("text", "") for row in rows if row.get("event") == "token")
    perf_events = [row for row in rows if row.get("event") == "perf"]
    require("." not in token_text, f"stop string leaked into JSONL text: {token_text}")
    require(len(perf_events) == 1, "stop JSONL did not emit exactly one perf event")
    require(0 < perf_events[0].get("decode_tokens", 0) < steps, "stop JSONL run did not stop early")
    return {
        "text": text_result["stdout"].strip(),
        "text_decode_tokens": text_perf["decode_tokens"],
        "jsonl_decode_tokens": perf_events[0]["decode_tokens"],
    }


def test_logit_bias(rina_infer, model, prompt, ctx, cwd, env, timeout):
    result = run(
        [
            str(rina_infer), "--gguf", "--model", str(model),
            "--prompt", prompt, "--steps", "1", "--ctx", str(ctx),
            "--logit-bias", "13:80.0", "--print-token-ids", "--quiet",
        ],
        cwd, env, timeout, "logit_bias",
    )
    check_command_ok(result)
    token_lines = [line.strip() for line in result["stdout"].splitlines() if line.strip().startswith("tokens:")]
    require(len(token_lines) == 1, "logit bias run did not emit one token line")
    token_ids = [int(x) for x in re.findall(r"-?\d+", token_lines[0])]
    require(token_ids == [13], f"logit bias did not force token 13: {token_ids}")
    perf = parse_prefixed_json(result["stderr"], "RINA_INFER_PERF")
    require(perf is not None, "missing RINA_INFER_PERF in logit bias run")
    return {
        "tokens": token_ids,
        "perf": perf,
    }


def test_llama_sampler_backend(rina_infer, model, prompt, ctx, cwd, env, timeout):
    text_result = run(
        [
            str(rina_infer), "--gguf", "--model", str(model),
            "--prompt", prompt, "--steps", "4", "--ctx", str(ctx),
            "--sampler-backend", "llama", "--quiet",
        ],
        cwd, env, timeout, "llama_sampler_text",
    )
    check_command_ok(text_result)
    require(text_result["stdout"].strip(), "llama sampler backend text output is empty")
    text_perf = parse_prefixed_json(text_result["stderr"], "RINA_INFER_PERF")
    require(text_perf is not None, "missing RINA_INFER_PERF in llama sampler text run")
    require(text_perf.get("decode_tokens") == 4, "llama sampler text decode_tokens mismatch")

    bias_result = run(
        [
            str(rina_infer), "--gguf", "--model", str(model),
            "--prompt", prompt, "--steps", "1", "--ctx", str(ctx),
            "--sampler-backend", "llama", "--logit-bias", "13:80.0",
            "--print-token-ids", "--quiet",
        ],
        cwd, env, timeout, "llama_sampler_logit_bias",
    )
    check_command_ok(bias_result)
    token_lines = [line.strip() for line in bias_result["stdout"].splitlines() if line.strip().startswith("tokens:")]
    require(len(token_lines) == 1, "llama sampler logit bias did not emit one token line")
    token_ids = [int(x) for x in re.findall(r"-?\d+", token_lines[0])]
    require(token_ids == [13], f"llama sampler logit bias did not force token 13: {token_ids}")
    return {
        "text": text_result["stdout"].strip(),
        "tokens": token_ids,
    }


def test_compare_gate(repo, model, text_path, args, env, timeout):
    compare = repo / "rina-engine/tools/compare_gguf_runtime.py"
    cmd = [
        sys.executable,
        str(compare),
        "--model", str(model),
        "--text", str(text_path),
        "--context", str(args.compare_context),
        "--stride", str(args.compare_stride),
        "--infer-steps", str(args.compare_infer_steps),
        "--bench-prompt", str(args.bench_prompt),
        "--bench-reps", str(args.bench_reps),
        "--tolerance", str(args.tolerance),
        "--perf-retries", str(args.perf_retries),
    ]
    if args.min_prompt_ratio > 0.0:
        cmd.extend(["--min-prompt-ratio", str(args.min_prompt_ratio)])
    if args.min_decode_ratio > 0.0:
        cmd.extend(["--min-decode-ratio", str(args.min_decode_ratio)])
    result = run(
        cmd,
        repo / "rina-engine",
        env,
        timeout,
        "compare_gate",
    )
    check_command_ok(result)
    summary = json.loads(result["stdout"])
    require(summary.get("ok") is True, f"compare gate returned ok=false: {result['stdout']}")
    require(summary.get("rina_detokenize_smoke") is True, "compare gate detokenize smoke failed")
    require(summary.get("relative_error", 1.0) <= args.tolerance, "PPL relative error exceeds tolerance")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run complete GGUF Phase 1 text output tests.")
    parser.add_argument("--model", default="/home/aquama/Development/llama.cpp/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--infer-steps", type=int, default=8)
    parser.add_argument("--token-id-steps", type=int, default=4)
    parser.add_argument("--jsonl-steps", type=int, default=4)
    parser.add_argument("--sampler-steps", type=int, default=16)
    parser.add_argument("--compare-context", type=int, default=64)
    parser.add_argument("--compare-stride", type=int, default=64)
    parser.add_argument("--compare-infer-steps", type=int, default=8)
    parser.add_argument("--bench-prompt", type=int, default=128)
    parser.add_argument("--bench-reps", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--min-prompt-ratio", type=float, default=0.0)
    parser.add_argument("--min-decode-ratio", type=float, default=0.0)
    parser.add_argument("--perf-retries", type=int, default=1)
    parser.add_argument("--smoke-repeats", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--cuda-bin", default="/home/aquama/miniconda3/envs/natalia/bin")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    rina_dir = repo / "rina-engine"
    model = Path(args.model).resolve()
    rina_infer = rina_dir / "build/rina_infer"
    eval_ppl = rina_dir / "build/eval_ppl"
    smoke_text = Path("/tmp/kilo/rina-gguf-phase1-smoke.txt")

    require(model.exists(), f"model does not exist: {model}")
    env = os.environ.copy()
    env["PATH"] = f"{args.cuda_bin}:{env.get('PATH', '')}"

    if not args.skip_build:
        for target in ("rina_infer", "eval_ppl"):
            build = run(
                ["cmake", "--build", "build", "--target", target],
                rina_dir,
                env,
                args.timeout,
                f"build_{target}",
            )
            check_command_ok(build)

    require(rina_infer.exists(), f"missing rina_infer binary: {rina_infer}")
    require(eval_ppl.exists(), f"missing eval_ppl binary: {eval_ppl}")
    # Keep enough text for at least two full PPL windows at the requested context.
    smoke_repeats = max(args.smoke_repeats, (args.compare_context * 3 + 31) // 32)
    write_smoke_text(smoke_text, smoke_repeats)

    default_text = test_default_text(rina_infer, model, args.prompt, args.infer_steps, args.ctx, rina_dir, env, args.timeout)
    output_modes = test_output_modes(rina_infer, model, args.prompt, args.infer_steps, args.ctx, rina_dir, env, args.timeout)
    token_ids = test_token_ids(rina_infer, model, args.prompt, args.token_id_steps, args.ctx, rina_dir, env, args.timeout)
    jsonl = test_jsonl(rina_infer, model, args.prompt, args.jsonl_steps, args.ctx, rina_dir, env, args.timeout)
    sampler_determinism = test_sampler_determinism(rina_infer, model, args.prompt, args.sampler_steps, args.ctx, rina_dir, env, args.timeout)
    sampler_params = test_sampler_params(rina_infer, model, args.prompt, args.sampler_steps, args.ctx, rina_dir, env, args.timeout)
    stop_strings = test_stop_strings(rina_infer, model, args.prompt, args.sampler_steps, args.ctx, rina_dir, env, args.timeout)
    logit_bias = test_logit_bias(rina_infer, model, args.prompt, args.ctx, rina_dir, env, args.timeout)
    llama_sampler_backend = test_llama_sampler_backend(rina_infer, model, args.prompt, args.ctx, rina_dir, env, args.timeout)
    compare = test_compare_gate(repo, model, smoke_text, args, env, args.timeout)

    summary = {
        "ok": True,
        "model": str(model),
        "smoke_text": str(smoke_text),
        "default_text": default_text,
        "output_modes": output_modes,
        "print_token_ids": token_ids,
        "jsonl": {
            "token_events": len([row for row in jsonl["events"] if row.get("event") == "token"]),
            "perf_events": len([row for row in jsonl["events"] if row.get("event") == "perf"]),
            "perf": jsonl["perf"],
        },
        "sampler_determinism": sampler_determinism,
        "sampler_params": sampler_params,
        "stop_strings": stop_strings,
        "logit_bias": logit_bias,
        "llama_sampler_backend": llama_sampler_backend,
        "compare_gate": compare,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        raise SystemExit(1)

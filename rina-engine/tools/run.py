#!/usr/bin/env python3
"""RINA Engine — standard inference runner with tokenizer decoding.
Usage:  python3 tools/run.py --model /tmp/model_a_v2.rinn --ids "hello world"
"""
import subprocess, sys, argparse

def main():
    ap = argparse.ArgumentParser(description='RINA Engine inference')
    ap.add_argument('--model', required=True, help='Path to .rinn model')
    ap.add_argument('--ids', default='128000',
                    help='Space-separated token IDs (default: BOS only)')
    ap.add_argument('--prompt', type=str, default=None,
                    help='Text prompt (overrides --ids, uses tokenizer to encode)')
    ap.add_argument('--steps', type=int, default=32, help='Tokens to generate')
    ap.add_argument('--temp', type=float, default=0, help='Temperature (0=greedy)')
    ap.add_argument('--topk', type=int, default=40)
    ap.add_argument('--topp', type=float, default=0.9)
    ap.add_argument('--rep', type=float, default=1.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tok', default='models/teacher/LLM-Research/Llama-3___2-1B-Instruct',
                    help='HuggingFace tokenizer path')
    ap.add_argument('--pytorch', action='store_true', help='Use PyTorch hybrid mode')
    ap.add_argument('--raw', action='store_true', help='Show raw token IDs only')
    args = ap.parse_args()

    # Load tokenizer for text prompts (always needed for decoding)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tok, local_files_only=True)

    # Convert text prompt to token IDs if provided
    if args.prompt is not None:
        encoded = tok.encode(args.prompt)
        ids_str = ' '.join(str(i) for i in encoded)
    else:
        ids_str = args.ids

    # Build engine command
    cmd = ['/home/aquama/Development/RINA_Project/rina-engine/build/rina_infer',
           '--model', args.model, '--ids', ids_str,
           '--steps', str(args.steps)]
    if args.temp > 0:
        cmd += ['--temp', str(args.temp), '--topk', str(args.topk),
                '--topp', str(args.topp), '--rep', str(args.rep)]
    if args.seed and args.temp > 0:
        cmd += ['--seed', str(args.seed)]
    if args.pytorch:
        cmd += ['--pytorch']

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f'Error: {proc.stderr}', file=sys.stderr)
        sys.exit(1)

    tokens = [int(x) for x in proc.stdout.strip().split()]
    if args.raw:
        print(' '.join(str(t) for t in tokens))
        return

    # Decode
    prompt_text = tok.decode([int(x) for x in ids_str.split()])
    gen_text = tok.decode(tokens)
    print(f'\nPrompt: {prompt_text}')
    print(f'Output: {gen_text}')

if __name__ == '__main__':
    main()

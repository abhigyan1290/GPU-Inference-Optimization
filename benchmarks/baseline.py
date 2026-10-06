"""Synchronized, fixed-work CUDA generation benchmark."""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoConfig, set_seed

from gpu_inference_lab.environment import collect_environment, require_cuda
from gpu_inference_lab.generation import generate, prompt_ids
from gpu_inference_lab.metrics import summarize
from gpu_inference_lab.model import DEFAULT_MODEL, load_model


def positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--batch-size", type=positive, default=1)
    parser.add_argument("--prompt-length", type=positive, default=512)
    parser.add_argument("--output-length", type=positive, default=128)
    parser.add_argument("--warmup", type=positive, default=3)
    parser.add_argument("--runs", type=positive, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=Path("results/raw"))
    args = parser.parse_args()
    require_cuda()
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("This GPU does not support bfloat16")
    set_seed(args.seed)
    metadata = collect_environment()
    config = AutoConfig.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    revision = getattr(config, "_commit_hash", None) or args.revision
    context = getattr(config, "max_position_embeddings", None)
    if context and args.prompt_length + args.output_length > context:
        parser.error(f"Prompt plus output exceeds model context ({context})")
    model, tokenizer = load_model(args.model, revision, getattr(torch, args.dtype))
    inputs_cpu = prompt_ids(tokenizer, args.prompt_length, args.batch_size)
    inputs = inputs_cpu.to("cuda")
    mask = torch.ones_like(inputs)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    latencies, samples = [], []
    for iteration in range(args.warmup + args.runs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = generate(model, inputs, mask, args.output_length, pad)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        if output.shape != (args.batch_size, args.prompt_length + args.output_length):
            raise RuntimeError("Generation did not perform the requested fixed workload")
        if not torch.equal(output[:, : args.prompt_length], inputs):
            raise RuntimeError("Generation changed the prompt prefix")
        if iteration >= args.warmup:
            latencies.append(elapsed)
            samples.append(
                {
                    "latency_s": elapsed,
                    "generated_tokens": output.numel() - inputs.numel(),
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                }
            )
        last_tokens = output[:, args.prompt_length :].cpu().tolist()
        del output  # Do not retain the preceding output tensor in the next memory measurement.
    timestamp = datetime.now(UTC)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "."], capture_output=True, text=True, check=False
    )
    result = {
        "schema_version": 1,
        "timestamp_utc": timestamp.isoformat(),
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "resolved_model_revision": revision,
        "environment": metadata,
        "git_commit": git.stdout.strip() or None,
        "git_status": status.stdout,
        "attention_implementation": "eager",
        "use_cache": True,
        "do_sample": False,
        "eos_stopping": False,
        "prompt_tokens_per_sequence": inputs.shape[1],
        "prompt_sha256": hashlib.sha256(inputs_cpu.numpy().tobytes()).hexdigest(),
        "generated_tokens_per_sequence": args.output_length,
        "samples": samples,
        "summary": summarize(latencies, args.batch_size * args.output_length),
        "ttft_s": None,
        "tpot_s": None,
        "token_metrics_note": "TTFT/TPOT not instrumented; latency includes prefill and decode.",
        "last_generated_token_ids": last_tokens,
        "sample_text": tokenizer.decode(last_tokens[0], skip_special_tokens=True),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    path = args.results_dir / (timestamp.strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print(f"Sample: {result['sample_text']}\nResults: {path}")


if __name__ == "__main__":
    main()

# Workload conventions

The CLI is the single source of configuration; JSON results record its resolved arguments.

Smoke: --batch-size 1 --prompt-length 64 --output-length 16 --warmup 1 --runs 3

Initial baseline: --batch-size 1 --prompt-length 512 --output-length 128 --warmup 3 --runs 10

These are presets to copy into commands, not another configuration parser. Do not run workload sweeps until Phase 2 is authorized.

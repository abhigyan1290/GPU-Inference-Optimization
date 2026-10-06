# Engineering rules

- Never fabricate hardware, profiler data, benchmark results, or speedups.
- Make performance claims only with measured evidence.
- Establish correctness before measuring performance.
- Follow hypothesis → implementation → correctness test → benchmark → profile → interpretation.
- Do not silently change methodology or several experimental variables at once.
- Explain nontrivial GPU/performance changes and record meaningful decisions in docs/DECISIONS.md.
- Prefer small, understandable patches; do not optimize unless the experiment calls for it.
- Add dependencies only for a concrete engineering need, never merely because AI suggested them.
- Preserve raw experimental outputs locally; intentionally select any artifacts committed to Git.
- Compare future low-level code against a trusted reference.
- Use uv sync --locked, uv run ruff check ., uv run ruff format --check ., and uv run pytest.
- Never install a Linux NVIDIA display driver in WSL. Keep vLLM in a separate future environment.
- Do not start Phase 2 or add optimizations without a new request.

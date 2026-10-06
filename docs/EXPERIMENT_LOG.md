# Experiment log

A smoke run validates execution, not a performance improvement. Keep exact raw-file references and limitations in every real experiment.

## Template

Experiment:
Date:
Question:
Hypothesis:
Configuration:
Independent variable:
Controlled variables:
Metrics:
Procedure:
Results:
Interpretation:
Limitations:
Next step:

## Phase 0/1 smoke validation

Experiment: GPU baseline harness smoke test
Date: 2026-10-06T21:47:25.523870+00:00
Question: Does the locked environment execute fixed-work GPU generation and save interpretable results?
Hypothesis: The small FP16 baseline fits this GPU and generates exactly the requested tokens.
Configuration: SmolLM2-135M, batch 1, prompt 64, output 16, warm-up 1, runs 3; exact software and revision in `results/raw/20261006T214725523870Z.json`.
Independent variable: None; validation only.
Controlled variables: Model, dtype, input IDs, generation settings, token counts.
Metrics: Synchronized generation wall time, generated tokens, allocator memory.
Procedure: Run environment arithmetic check, CPU correctness suite, then the smoke CLI and inspect JSON.
Results: 12 tests passed; CUDA arithmetic and all three generation runs succeeded with 16 tokens each. Median 0.4688 s; aggregate 33.19 output tokens/s.
Interpretation: The Phase 1 harness works on this machine.
Limitations: Three samples, one warm-up, synthetic repeated prose, uncontrolled laptop clocks; no meaningful tail-latency or speedup claim.
Next step: Developer reviews methodology and learning questions before authorizing Phase 2.


## Gemma baseline validation

Experiment: Verify the new Gemma baseline executes on the local GPU.
Date: 2026-10-06T23:12:37.826190+00:00
Question: Does Gemma 3 1B execute the fixed-work benchmark successfully in BF16?
Hypothesis: The short batch-one workload fits the RTX 4060's memory.
Configuration: `google/gemma-3-1b-pt`, revision `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`; prompt 64, output 16, batch 1, warm-up 3, runs 5, BF16, seed 0.
Independent variable: None within the run; validation only. Relative to the historical SmolLM2 smoke run, model, dtype, warm-up count, and repetition count differ.
Controlled variables: Input IDs and generation settings within the run.
Metrics: Synchronized generation latency, output-token throughput, PyTorch peak allocated/reserved bytes.
Procedure: User ran the smoke benchmark; agent inspected saved JSON and checked counts and arithmetic afterward. No new inference run was performed for this entry.
Results: Five runs each produced 16 tokens. Median 1.166504 s; range 1.062658–1.762753 s; throughput 12.0019 tokens/s. Raw evidence: `results/raw/20261006T231237826190Z.json`.
Interpretation: Model access and GPU execution are validated for this workload.
Limitations: Synthetic prompt without added special tokens; substantial latency variation; five samples; laptop conditions not controlled; recorded source tree dirty; no Gemma-specific cache/reference comparison. Existing 12-test suite was validated separately before this run and uses a tiny Llama for reference-generation checks.
Next step: Commit the intended baseline changes, add Gemma-specific correctness checks and a separate normal-tokenization smoke check, then establish repeatability before Phase 2 sweeps.

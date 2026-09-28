# MP1 Student Model Report

## Final result

The final validation-selected checkpoint is `runs/rope-calibrated/best.pt`.
Validation BPB is **1.4744189553** and frozen full-test CPU FP32 BPB is
**1.4935076045**. The older 1.5627123602 result belongs to a historical
large-cache model and is not the final submission result.

The model is a six-block rotary causal Transformer with RMSNorm, width 288,
six attention heads, FFN hidden size 768, bias-free projections and tied
token/output embeddings. It has 7,155,360 parameters. Training used only the
supplied WikiText-2 training split, seed 17, BF16 AdamW, peak learning rate
0.0008, 24,000 updates, residual dropout 0.2, embedding dropout 0.1, and
averaged validation snapshots from step 12,000.

Inference uses temperature 1.2 and mixes the neural distribution with a
strictly causal within-window repetition distribution using alpha 0.05, cosine
temperature 15 and suffix-match bonus 2. Each cache key at position j can only
affect later positions; its value is the already observed token at j+1. Cache
state is temporary and resets for every independent 256-token window.

## Compliance measurements

The evaluator and tokenizer are unchanged. The test contains 428,405 targets.
CPU FP32 scoring on four threads took **31.24945 seconds**. The retained
same-machine baseline timing is 11.88288 seconds, giving **2.63x**, below the
5x limit. Peak process RSS was **1.84972 GiB**, below 4 GiB. The checkpoint is
**28,631,719 bytes (27.30 MiB)**, below the 64 MiB inference-asset limit.

The actual checkpoint passed finite-probability, normalization, prefix
equivalence, future-input perturbation, example-independence and window-reset
checks. The retained test suite passes 11 tests. Evidence is in
`runs/rope-calibrated/causality.json`, `test_cpu_fp32.json`,
`test_cpu_fp32.resources.json` and `selection.json`.

## Validation search and ablation

| Candidate | Validation BPB | Status |
|---|---:|---|
| Original six-layer cached GPT | 1.5449619448 | Historical starting point |
| Six-layer continuation and averaging | 1.5292658663 | Rejected |
| Eight-layer calibration | 1.4966693223 | Rejected by final candidate |
| RoPE model before calibration | 1.5350178418 | Calibration input |
| Final RoPE calibration | **1.4744189553** | Selected |

All settings were selected on validation. The final test score was measured
only after freezing the method; no test score was used for tuning.

## Reproduction

From `code/`, install requirements and run:

```bash
python -m unittest discover -s tests -v
python evaluate.py --checkpoint runs/rope-calibrated/best.pt --device cpu --precision fp32 --threads 4 --split test --output runs/rope-calibrated/reproduced_test.json
```

The final implementation is `student_rope.py`, using `student_cache.py`. The
baseline remains in `model.py`. The complete bundle is
`deliverables/MP1_final_submission.zip`; it contains the matching checkpoint
as `submission/best.pt`, source, data, tokenizer, tests, reports and evidence.

## AI assistance disclosure

AI assistance was used to inspect the starter code, implement and review the
rotary model and causal cache, run validation experiments, measure resources,
check causality and draft the report. Training used no pretrained weights,
external text, retrieval database or test answers.

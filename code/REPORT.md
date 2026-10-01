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
An adjacent four-thread CPU FP32 timing pair measured the baseline at
**7.24619 seconds** and the final model at **27.67110 seconds**, a ratio of
**3.82x**, below the 5x limit. A separate final scoring pass took 31.24945
seconds; timings vary with system load. Peak process RSS was **1.84972 GiB**,
below 4 GiB. The checkpoint is
**28,631,719 bytes (27.30 MiB)**, below the 64 MiB inference-asset limit.
The paired baseline is the seed-17, 1,200-update FP32 baseline checkpoint
`runs/baseline-timing/best.pt`, SHA256
`3ac1501ddcdec3af1a9dca08eaa11983793745104c978b77267370801ce94d9d`.
Both paired score files use protocol `7506-mp1-wt2-v2`, full test, CPU FP32,
four threads, the same tokenizer and the unchanged evaluator. The old 11.88288
second baseline figure in an earlier report version had no retained source
measurement and is superseded by this traceable pair.

The actual checkpoint passed finite-probability, normalization, prefix
equivalence, future-input perturbation, example-independence and window-reset
checks. The retained test suite passes 11 tests. Evidence is in
`runs/rope-calibrated/causality.json`, `test_cpu_fp32.json`,
`test_cpu_fp32.resources.json`, `selection.json` and the paired `cpu_pair_*.json`
measurements with their resource companions.

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

The cache ablation uses the same final weights and validation set. With
temperature 1.2, the best zero-cache setting scores 1.5043930520 BPB; the
selected cache scores 1.4744189553, a 0.0299740967 BPB reduction. The complete
725-setting grid is retained in `runs/rope-calibrated/selection.json`.

At equal training budgets of 9,830,400 targets, the baseline validation BPB is
2.0710836385 and the 1,200-step RoPE recipe is 1.7414554194. This is a complete
recipe comparison: baseline training was FP32 CPU while the RoPE control used
BF16 CUDA. Search target counts, times and limitations are detailed in
`EXPERIMENT_RECORD.md`.

## Reproduction

From `code/`, install requirements and run:

```bash
python -m unittest discover -s tests -v
python train.py --implementation student_rope --config configs/student-rope.json --device cuda --precision bf16 --seed 17 --steps 24000 --eval-every 1200 --average-start 12000 --lr 0.0008 --run-dir runs/repro-rope
python select_cache.py --checkpoint runs/repro-rope/best.pt --run-dir runs/repro-rope-calibrated
python evaluate.py --checkpoint runs/repro-rope-calibrated/best.pt --device cpu --precision fp32 --threads 4 --split test --output runs/repro-rope-calibrated/test_cpu_fp32.json
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

# Experiment record

## Final model

The retained checkpoint is `runs/rope-calibrated/best.pt`.

- Implementation: `student_rope.py` with the causal cache in `student_cache.py`
- Architecture: six rotary RMSNorm blocks, width 288, six heads, FFN hidden 768, bias-free projections, tied token/output embeddings
- Parameters: 7,155,360
- Context/vocabulary: 256 / 2048
- Training: seed 17, BF16, AdamW, peak learning rate 0.0008, 24,000 updates, residual dropout 0.2, embedding dropout 0.1, averaged snapshots from step 12,000
- Inference calibration: temperature 1.2, cache alpha 0.05, theta 15, suffix-match bonus 2
- Validation BPB: 1.4744189553
- Test BPB: 1.4935076045
- Adjacent CPU FP32 timing pair: baseline 7.24619 s, final 27.67110 s, ratio 3.82x; a separate final-only pass took 31.24945 s
- Baseline checkpoint: `runs/baseline-timing/best.pt`, SHA256 `3ac1501ddcdec3af1a9dca08eaa11983793745104c978b77267370801ce94d9d`
- Baseline full-test BPB: 2.1012604381; validation BPB: 2.0710836385; 1,200 updates / 9,830,400 targets; baseline training metrics and both baseline checkpoints are retained in `runs/baseline-timing/`
- Baseline timing was measured using `best.pt` (hash above). `metrics.json`'s `checkpoint_sha256` identifies `checkpoint.pt` (last-step file), not `best.pt`; both contain identical model tensors, verified directly, and both correspond to step 1,200.
- Peak process RSS: 1.84972 GiB
- Checkpoint: 28,631,719 bytes (27.30 MiB)
- Checkpoint SHA256: d6825c653df361d4dc171047de6e04e76d55175165f244543672632235532d1c

## Validation search history

| Candidate | Validation BPB | Outcome |
|---|---:|---|
| Original six-layer cached GPT | 1.5449619448 | Historical starting point |
| Six-layer continuation and averaging | 1.5292658663 | Rejected |
| Six-layer temperature/cache calibration | 1.5188264986 | Rejected |
| Eight-layer continuation and averaging | 1.5160812962 | Rejected |
| Eight-layer calibration | 1.4966693223 | Rejected by final RoPE candidate |
| RoPE model before calibration | 1.5350178418 | Calibration candidate |
| Final RoPE calibration | 1.4744189553 | Retained |

## GUIDE experimental evidence and search cost

Equal-target comparison: the plain baseline and RoPE recipe each processed
9,830,400 targets (1,200 x 32 x 256). Baseline validation was 2.0710836385;
the RoPE 1,200-step validation was 1.7414554194. The baseline used FP32 CPU
training and the RoPE control used BF16 CUDA training with its stated recipe,
so this compares complete training recipes at equal target count; it does not
isolate architecture or precision. The control run's metrics were summarized
here and its temporary checkpoint was removed.

Key mechanism ablation uses the exact final RoPE weights and full validation
set. With temperature fixed at the selected 1.2, the best zero-cache setting
has BPB 1.5043930520; the selected cache mixture gives 1.4744189553, a
0.0299740967 BPB reduction. The complete 725-setting validation grid is in
`runs/rope-calibrated/selection.json` and can be regenerated with
`select_cache.py`. A fresh rerun of that selector reproduced the selected
settings and BPB to <2e-13; its measured validation sweep took 13.59 seconds.

Training/search budgets for the renewed search: baseline timing run 9,830,400
targets / 341.86 CPU seconds; equal-target RoPE control 9,830,400 / 67.08 GPU
training seconds; final RoPE ancestry 196,608,000 / about 1,588 seconds on the
RTX 2000 Ada laptop GPU. Rejected continuations processed 49,152,000 targets
(six-layer), 98,304,000 (eight-layer growth), and 24,576,000 (joint cache-loss
training). These are search costs in addition to the selected model's
ancestry. Cache calibration evaluates validation only; no test-based tuning
was performed. Search costs from earlier experiments are summarized in the
earlier report; detailed per-run JSON from that earlier phase was removed when
the user requested a minimal final-plus-baseline folder.

All settings were selected on validation. No test score was used for tuning.
The retained 11-test suite passes; the actual checkpoint passed causal,
normalization, prefix-equivalence, future-perturbation, example-independence
and window-reset checks. The final test was run only after selection and freeze.

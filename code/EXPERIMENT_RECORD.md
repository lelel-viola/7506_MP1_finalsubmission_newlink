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
- Test CPU FP32 time: 31.24945 s; retained baseline: 11.88288 s; ratio: 2.63x
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

All settings were selected on validation. No test score was used for tuning.
The final checkpoint passed the 31-test suite and the actual-checkpoint causal,
normalization, prefix-equivalence, future-perturbation, example-independence
and window-reset checks. The final test was run only after selection and freeze.

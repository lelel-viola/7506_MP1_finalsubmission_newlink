# MP1 Student Model Report

> **Active validation-only search (2026-09-28):** the current best candidate is
> `runs/rope-calibrated/best.pt`, validation BPB **1.4744189553**.
> This is an eight-layer continuation with averaged weights, temperature 1.15,
> cache alpha 0.05, theta 15 and causal suffix-match bonus 2. No new test score
> or resource-compliance claim is available. At the user's request, CPU resource
> measurements and final test scoring are deferred while optimization continues.
> The report below describes the previous six-layer submission; its complete
> The compact historical comparison is preserved in `EXPERIMENT_RECORD.md`.

> **Current frozen candidate after validation selection:**
> `runs/rope-calibrated/best.pt`. It was selected using validation BPB
> **1.4744189553** and then frozen before test scoring. Its full CPU FP32 test
> BPB is **1.4935076045**. Test scoring took **31.24945 s** on four CPU
> threads; the retained baseline timing is 11.88288 s, giving 2.63x. Peak
> process RAM was **1.84972 GiB** and the checkpoint is **28,631,719 bytes**
> (27.30 MiB), both within the guide limits. Causality and normalization are
> recorded in `runs/rope-calibrated/causality.json`; the test result and resource
> report are `runs/rope-calibrated/test_cpu_fp32.json` and its `.resources.json`.

## Method

The final model is a six-block, width-256 causal GPT with four attention heads,
SwiGLU feed-forward layers, tied embeddings and 5,725,196 trainable parameters.
Residual-branch dropout is 0.1 during training and disabled for evaluation.
AdamW uses weight decay 0.1, peak LR 0.0008, 100-step warmup, cosine decay,
gradient clipping 1.0, batch size 32 and seed 17. Training runs for 12,000
updates in BF16; full FP32 validation every 1,200 updates selects step 10,800.

The final predictor combines 95% neural probability with 5% within-window
contextual repetition probability. At position t, key j is the normalized
hidden representation through input j, and its value is the already observed
input j+1. Only j<t is allowed. We softmax 10 times the cosine similarity to
the current representation and sum weights by the continuation token. The
first position uses the neural predictor alone. All temporary memory is local
to the call. This exploits repeated vocabulary without a stored retrieval
 database. Source: student_rope.py and student_cache.py.

The model uses only the supplied WikiText-2 training split and tokenizer. It
uses independent causal windows of 256 tokens and carries no state across
windows or examples. It does not access future targets, external training
data, pretrained weights or cached validation/test answers. The fixed data,
tokenizer, common.py, model.py and evaluate.py retain their supplied hashes.

## Measurements

All runs used seed 17 and batch size 32. BPB is total target negative log
probability in nats divided by ln(2) and the split's raw UTF-8 bytes. Lower is
better. Validation alone selected checkpoints and cache hyperparameters.
An earlier candidate was tested before the user requested further development;
that historical test result was excluded from the renewed search. The final
predictor was frozen before its own full-test CPU evaluation.

| Model | Training updates searched | Validation BPB |
|---|---:|---:|
| Plain baseline, measured | 1,200 | 2.07108 |
| Existing five-layer width-144 student | 1,200 | 1.94221 |
| Same medium student, whole-block dropout 0.1 | 6,000 | 1.78483 |
| Same medium student, no dropout | 6,000 | 1.70100 |
| Large student, cache off (selected step 10,800) | 12,000 | 1.58755 |
| Same large checkpoint, cache on | 12,000 | 1.54496 |

Final full CPU FP32 test BPB is **1.5627123602**, versus measured baseline
2.1012568157 (25.63% lower). All 428,405 targets are scored. CPU validation
reproduces GPU selection within 0.0000001 BPB. Final artifacts are
runs/rope-calibrated/test_cpu_fp32.json and its .resources.json companion.
Both evaluations use four CPU threads on the same machine, with training
stopped. The final adjacent pair took 11.88288 s (baseline) and 40.19179 s
(student), or **3.38x**, below 5x. Peak student process RAM was **1.84961 GiB**,
below 4 GiB. The checkpoint is **25,022,644 bytes (23.86 MiB)**; its few source
modules add far less than 1 MiB, below the 64 MiB inference-asset budget.
Timing is sensitive to laptop load and thermal state: earlier baseline runs
ranged from 7.58 to 12.11 seconds. All measurements are retained; compliance
here is the final matched local pair, not a guarantee for every environment.
Probability mixing was implemented equivalently in probability space and
then as direct scatter additions to reduce CPU overhead; weights and mixture
settings remained fixed. Repeated test scoring checks runtime and numerical
equivalence, with BPB differences below 0.00000002.
measure_evaluation.py measures peak process working-set RAM including loading;
the evaluator's zero-valued CPU GPU-allocation fields are not RAM evidence.

Equal-target comparison: plain baseline and the existing medium student each
process 9,830,400 targets. This compares the complete architecture change,
including size and initialization, not SwiGLU in isolation. Additionally the
large run's 1,200-step validation is 1.81934 at the same target count, but its
precision and learning-rate schedule differ, so this is a recipe comparison.

Key mechanism ablation: turning off cache on the exact same large weights
changes validation BPB from 1.54496 to 1.58755, a 0.04259 BPB improvement with
no extra training targets. Nine nonzero (alpha,theta) settings and alpha=0
were evaluated in one validation forward pass; all results are retained in
runs/rope-calibrated/selection.json. The grid uses alpha={0.05,0.1,0.2} and
theta={10,20,40}. No test scores enter this selection.

Negative result: whole-block dropout 0.1 worsens the medium model at equal
6,000-step budgets. Residual-branch dropout in the larger recipe is a different
operation; its independent benefit has not been isolated. Larger capacity and
longer training explain part of the gain, but the experiment does not identify
each contribution. This is a single-seed study, not a statistical significance
claim. The original 1.4 target has not been achieved.

Search cost: the four new training runs used 206,438,400 targets and about
1,130.89 GPU training seconds in total (excluding loading/validation). The
pre-existing medium run adds 9,830,400 targets and 72.09 seconds. The final
large run alone searched 98,304,000 targets; the selected checkpoint's ancestry
is 88,473,600 targets. Cache calibration took 5.40 seconds excluding loading.
All per-run costs and validation curves remain in runs/*/metrics.json.

## Reproduction

From `code/`, install dependencies and run:

```bash
python -m unittest discover -s tests -v
python train.py --implementation model --device cuda --precision fp32 --seed 17 --steps 1200 --run-dir runs/repro-baseline
python train.py --implementation student --config configs/student-large.json --device cuda --precision bf16 --seed 17 --steps 12000 --eval-every 1200 --lr 0.0008 --run-dir runs/repro-large
python tune_cache.py --checkpoint runs/repro-large/best.pt --run-dir runs/repro-cache --device cuda
python measure_evaluation.py --checkpoint runs/repro-cache/best.pt --device cpu --precision fp32 --threads 4 --split test --output runs/repro-cache/test_cpu_fp32.json
```

The frozen checkpoint is runs/rope-calibrated/best.pt. Direct scoring needs no
retraining. In the submission archive it is submission/best.pt. Install with
Python 3.12 and requirements.txt; training used PyTorch 2.7.1+cu126 on an RTX
2000 Ada Generation Laptop GPU. Installed NumPy is 2.5.3 and tokenizers 0.21.4.
CPU evaluation uses Windows 11 and the same PyTorch installation. The current
machine's interpreter is C:/anaconda/envs/mp1/python.exe.

The full original and extended causal/normalization/gradient tests pass
(18 tests). The cache test additionally compares every truncated prefix with
the corresponding full-window predictions and verifies numerical equivalence
to the log-space mixture. Each submitted file's SHA256 is in the archive's
BUNDLE_MANIFEST.json. Upload an immutable source snapshot and the checkpoint
bundle, enter the student ID and exact test BPB, and keep this report within
10 pages when exporting it. Those website/account actions are not performed
by the local experiment scripts.

## AI Assistance Disclosure

AI assistance was used to inspect the starter code, review the evaluation
contract, implement and test the SwiGLU model and causal contextual cache,
run validation experiments, measure resources, and draft this report.
The model was trained from scratch on the supplied data
without pretrained weights or external training text.

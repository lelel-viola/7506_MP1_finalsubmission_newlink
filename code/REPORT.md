# MP1 Student Model Report

## Method

The final model keeps the baseline causal GPT interface and replaces each GELU
MLP with a SwiGLU-style gated feed-forward network. Each token produces a value
projection and a SiLU gate; their elementwise product is projected back to the
model width. The final configuration uses five blocks, width 144 and four
attention heads. Token embeddings and output weights remain tied.

The model uses only the supplied WikiText-2 training split and tokenizer. It
uses independent causal windows of 256 tokens, has no evaluation-time memory,
and does not access future tokens, external data, retrieval tables or test
answers.

## Measurements

All runs used seed 17, batch size 32, 1,200 updates and 9,830,400 processed
training targets. Validation was used for selection; the test set was evaluated
only after freezing the configuration.

| Model | Parameters | Validation BPB | Test BPB |
|---|---:|---:|---:|
| Guide baseline, approximate | 1,088,256 | - | 2.10 |
| Final SwiGLU student | 1,690,714 | 1.94221 | **1.97385** |

The final test was run with CPU FP32 evaluation and produced 428,405 scored
targets. The measured evaluation time was 34.65 seconds on the local machine.
The guide's reference baseline scoring time is 5.92 seconds, so the final model's
local timing is approximately 5.85x that reference and should be remeasured on
the course reference environment before claiming full time-budget compliance.
Peak evaluation memory was below 4 GiB and the checkpoint is below the 64 MiB
inference-asset limit.

Ablation evidence: the earlier four-block SwiGLU student reached validation BPB
2.04601 with the same training target budget. Increasing capacity to five blocks
and width 144 reduced validation BPB to 1.94221 and test BPB to 1.97385. The
remaining ablation against the untouched `model.py` baseline should be run and
logged with the same seed and target count if a formal report is required.

## Reproduction

From `code/`, install dependencies and run:

```bash
python -m unittest discover -s tests -v
python train.py --implementation student --config configs/student-medium.json --device cpu --threads 4 --seed 17 --steps 1200 --eval-every 300 --run-dir runs/my-model
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --device cpu --precision fp32 --threads 4 --split validation
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

The matching frozen checkpoint from the reported run is
`runs/student-medium-1200/checkpoint.pt`.

## AI Assistance Disclosure

AI assistance was used to inspect the starter code, review the evaluation
contract, implement and test the SwiGLU student model, clean up experiments,
and draft this report. The model was trained from scratch on the supplied data
without pretrained weights or external training text.

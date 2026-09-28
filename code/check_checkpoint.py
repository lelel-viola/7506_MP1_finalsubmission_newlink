"""FP32 causal contract checks on the actual selected checkpoint."""
import argparse
import json
from pathlib import Path
import torch
from common import make_model, setup, sha


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    device, _ = setup('cpu', 'fp32', 4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(checkpoint['implementation'], checkpoint['config'], device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    torch.manual_seed(123)
    x = torch.randint(2048, (2, 256))
    full = model.predict_log_probs(x)
    assert full.shape == (2, 256, 2048) and torch.isfinite(full).all()
    # Long trained-model reductions can accumulate a few FP32 ulps; this is
    # still 100 times tighter than the fixed evaluator's 1e-3 contract.
    normalization_error = full.logsumexp(-1).abs().max().item()
    torch.testing.assert_close(full.logsumexp(-1), torch.zeros(2, 256), atol=1e-5, rtol=0)
    errors = {}
    for length in (1, 2, 7, 31, 128, 255):
        prefix = model.predict_log_probs(x[:, :length])
        torch.testing.assert_close(prefix, full[:, :length], atol=2e-5, rtol=1e-5)
        changed = x.clone()
        changed[:, length:] = (changed[:, length:]+123) % 2048
        changed_probs = model.predict_log_probs(changed)
        torch.testing.assert_close(changed_probs[:, :length], full[:, :length], atol=2e-5, rtol=1e-5)
        errors[str(length)] = (prefix-full[:, :length]).abs().max().item()
    alone = model.predict_log_probs(x[:1])
    torch.testing.assert_close(alone, full[:1], atol=2e-5, rtol=1e-5)
    model.predict_log_probs((x+37) % 2048)
    repeated = model.predict_log_probs(x)
    torch.testing.assert_close(repeated, full, atol=0, rtol=0)
    result = dict(checkpoint_sha256=sha(args.checkpoint), precision='fp32', device='cpu',
                  passed=True, normalization_max_abs_error=normalization_error,
                  checks=['finite', 'normalized', 'prefix equivalence', 'future perturbation',
                                       'example independence', 'window reset'], prefix_max_abs_error=errors)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Select the causal cache and temperature using validation only."""
import argparse
import itertools
import json
import math
from pathlib import Path
import time
import torch
from common import load_data, make_model, setup, sha, windows
from student_cache import cache_weights


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.run_dir.exists():
        parser.error('Choose a new output directory.')
    device, _ = setup('cuda', 'fp32', 4)
    parent = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, _ = make_model(parent['implementation'], parent['config'], device)
    model.load_state_dict(parent['model'])
    model.eval()
    tokens, byte_count = load_data()['validation']
    temperatures = (1., 1.05, 1.1, 1.15, 1.2)
    alphas = (.02, .05, .08, .12)
    thetas = (5., 10., 15., 20.)
    token_bonuses = (0., 2., 4.)
    suffix_bonuses = (0., 2., 4.)
    keys = [(t, 0., 0., 0., 0.) for t in temperatures]
    keys += list(itertools.product(temperatures, alphas, thetas,
                                   token_bonuses, suffix_bonuses))
    nll = dict.fromkeys(keys, 0.)
    started = time.perf_counter()
    for ids, targets in windows(tokens):
        ids, targets = ids.to(device), targets.to(device)
        features = model.features(ids)
        logits = model.head(features).float()
        gather_ids = targets.clamp_min(0).unsqueeze(-1)
        valid = targets != -100
        base_by_temp = {}
        for temperature in temperatures:
            base_by_temp[temperature] = (logits / temperature).log_softmax(-1).gather(
                -1, gather_ids).squeeze(-1)
            nll[(temperature, 0., 0., 0., 0.)] += (
                -base_by_temp[temperature][valid]).double().sum().item()
        for theta, token_bonus, suffix_bonus in itertools.product(
                thetas, token_bonuses, suffix_bonuses):
            weights, values = cache_weights(features, ids, theta,
                                            token_bonus, suffix_bonus)
            cache_target = (weights * (values[:, None, :] == gather_ids)).sum(-1)
            cache_log = cache_target.clamp_min(torch.finfo(torch.float32).tiny).log()
            for temperature in temperatures:
                base = base_by_temp[temperature]
                for alpha in alphas:
                    mixed = torch.logaddexp(base + math.log1p(-alpha),
                                            cache_log + math.log(alpha))
                    mixed[:, 0] = base[:, 0]
                    nll[(temperature, alpha, theta, token_bonus, suffix_bonus)] += (
                        -mixed[valid]).double().sum().item()
    rows = [dict(temperature=t, alpha=a, theta=h, token_bonus=b,
                 suffix_bonus=s, bpb=value / math.log(2) / byte_count)
            for (t, a, h, b, s), value in nll.items()]
    best = min(rows, key=lambda row: row['bpb'])
    selected = dict(parent)
    selected['config'] = dict(parent['config'], temperature=best['temperature'],
                              cache_alpha=best['alpha'], cache_theta=best['theta'],
                              cache_token_bonus=best['token_bonus'],
                              cache_suffix_bonus=best['suffix_bonus'])
    selected['validation_bpb'] = best['bpb']
    selected['parent_checkpoint_sha256'] = sha(args.checkpoint)
    args.run_dir.mkdir(parents=True)
    torch.save(selected, args.run_dir/'best.pt')
    report = dict(selection_split='validation', parent_checkpoint=str(args.checkpoint),
                  parent_sha256=sha(args.checkpoint), best=best, grid=rows,
                  seconds=time.perf_counter()-started)
    (args.run_dir/'selection.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

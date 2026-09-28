"""Strictly within-window contextual repetition model; no persistent memory."""
import torch
from torch.nn import functional as F


def cache_weights(features, ids, theta, token_bonus=0.0, suffix_bonus=0.0):
    batch, length = ids.shape
    normalized = F.normalize(features.float(), dim=-1)
    similarities = theta * (normalized @ normalized.transpose(-1, -2))
    if token_bonus or suffix_bonus:
        matches = ids[:, :, None] == ids[:, None, :]
        if token_bonus:
            similarities = similarities + token_bonus * matches
        if suffix_bonus:
            suffix = matches
            for _ in range(2, 5):
                previous = torch.zeros_like(suffix)
                previous[:, 1:, 1:] = suffix[:, :-1, :-1]
                suffix = previous & matches
                similarities = similarities + suffix_bonus * suffix
    allowed = torch.arange(length, device=ids.device)[None, :] < torch.arange(length, device=ids.device)[:, None]
    allowed[0, 0] = True
    weights = similarities.masked_fill(~allowed, float('-inf')).softmax(-1)
    values = torch.cat((ids[:, 1:], ids[:, -1:]), dim=1)
    return weights, values


def cache_probs(features, ids, theta, vocab, token_bonus=0.0, suffix_bonus=0.0):
    batch, length = ids.shape
    weights, values = cache_weights(features, ids, theta, token_bonus, suffix_bonus)
    probs = torch.zeros(batch, length, vocab, device=ids.device, dtype=torch.float32)
    return probs.scatter_add(-1, values[:, None, :].expand(batch, length, length), weights)


def mixed_log_probs(features, logits, ids, config):
    temperature = config.get('temperature', 1.0)
    if temperature <= 0:
        raise ValueError('temperature must be positive.')
    logits = logits.float() / temperature
    alpha = config.get('cache_alpha', 0.0)
    if not 0 <= alpha < 1:
        raise ValueError('cache_alpha must be in [0, 1).')
    if alpha == 0 or ids.shape[1] == 1:
        return F.log_softmax(logits, dim=-1)
    base = F.softmax(logits, dim=-1)
    weights, values = cache_weights(features, ids, config.get('cache_theta', 20.),
                                    config.get('cache_token_bonus', 0.0),
                                    config.get('cache_suffix_bonus', 0.0))
    base[:, 1:] *= 1-alpha
    weights[:, 0] = 0
    weights *= alpha
    base.scatter_add_(-1, values[:, None, :].expand(ids.shape[0], ids.shape[1], ids.shape[1]), weights)
    return base.clamp_min_(torch.finfo(base.dtype).tiny).log_()

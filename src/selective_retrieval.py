from contextlib import contextmanager
from types import MethodType
from unittest.mock import patch

import torch


def blank_only_process(wrapper, hidden_states, am_log_probs):
    if hidden_states.size(0) != 1 or am_log_probs.size(0) != 1:
        raise ValueError('Selective prototype only supports one utterance')
    if not wrapper.decode_skip_blank or wrapper.scale_lmbda or wrapper.recompute_dists:
        raise ValueError('Selective prototype requires frozen greedy configuration')
    wrapper.vocab_size = am_log_probs.shape[-1]
    selected = torch.nonzero(am_log_probs[0].argmax(-1) != 0, as_tuple=False).flatten()
    if selected.numel() == 0:
        return (am_log_probs.clone(), am_log_probs.new_empty((0, wrapper.k)),
                torch.empty((0, wrapper.k), dtype=torch.long, device=am_log_probs.device))
    queries = hidden_states.index_select(1, selected)
    distances, ids = wrapper.get_knns(queries)
    probabilities, labels = wrapper.knns_to_log_prob(ids, -distances)
    full_probabilities = torch.full_like(am_log_probs, -10000.)
    full_probabilities[0, selected] = probabilities
    fused = type(wrapper).interpolate(full_probabilities, am_log_probs,
                                      wrapper.lmbda, wrapper.decode_skip_blank)
    return fused, distances, labels


@contextmanager
def selective_mode(engine, enabled=False):
    if not enabled:
        yield
        return
    with patch.object(engine.wrapper, 'process', MethodType(blank_only_process, engine.wrapper)):
        yield

from contextlib import contextmanager
from types import MethodType

import torch


def device_allocation_aggregate(model, knns, neg_dists):
    probs = torch.nn.functional.softmax(neg_dists / model.knn_temperature, dim=-1)
    vals_at_knns = model.vals[knns].squeeze(-1)
    knn_log_probs = torch.full(
        size=vals_at_knns.shape[:-1] + (model.vocab_size,),
        fill_value=0.0,
        device=model.device,
    ).scatter_add(dim=-1, index=vals_at_knns, src=probs).log()
    knn_log_probs = torch.nan_to_num(knn_log_probs, nan=None, neginf=-10000.0)
    return knn_log_probs, vals_at_knns


@contextmanager
def aggregation_mode(model, enabled=False):
    if not enabled:
        yield
        return
    existed = 'knns_to_log_prob' in model.__dict__
    previous = model.__dict__.get('knns_to_log_prob')
    model.knns_to_log_prob = MethodType(device_allocation_aggregate, model)
    try:
        yield
    finally:
        if existed:
            model.knns_to_log_prob = previous
        else:
            del model.knns_to_log_prob

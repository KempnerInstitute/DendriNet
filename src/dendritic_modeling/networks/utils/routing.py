"""Feature-to-destination assignments shared by pathway and E/I routers."""

import torch
from torch.nn import functional as F


def routing_assignments(logits, temperature, *, dim=-1, hard=False, training=False):
    """Softmax over destinations, optionally one-hot with a straight-through VJP.

    Callers validate their configuration. The default is the original pathway
    router's soft assignment, with unchanged arithmetic and no random draws.
    Hard training uses ``discrete + soft - soft.detach()``, the expression of
    the reduced models it replays; a selected entry can then differ from one in
    its last bit (``discrete + (soft - soft.detach())`` would be exact).
    Evaluation uses exact one-hot entries. This is not Top-K contact selection
    within a row.
    """
    soft = torch.softmax(logits / temperature, dim=dim)
    if not hard:
        return soft
    discrete = F.one_hot(soft.argmax(dim=dim), num_classes=logits.shape[dim])
    discrete = discrete.movedim(-1, dim).to(soft.dtype)
    return discrete + soft - soft.detach() if training else discrete


def assignment_statistics(assignment, *, destination_dim):
    """Mean source entropy and mean squared departure from equal destination mass."""
    matrix = assignment.movedim(destination_dim, 0)
    # 1e-8 rounds to zero in float16, whose smallest normal number then serves
    # as the floor; other dtypes keep 1e-8.
    floor = max(1e-8, torch.finfo(matrix.dtype).tiny)
    entropy = -(matrix * torch.log(matrix.clamp_min(floor))).sum(dim=0).mean()
    mass = matrix.sum(dim=1) / matrix.shape[1]
    balance = ((mass - 1.0 / matrix.shape[0]) ** 2).mean()
    return entropy, balance

"""Feature-to-destination assignments shared by pathway and E/I routers."""

import torch
from torch.nn import functional as F

from dendritic_modeling.networks.functional import _check_operand


def routing_assignments(logits, temperature, *, dim=-1, hard=False, training=False):
    """Softmax over destinations, optionally one-hot with a straight-through VJP.

    Temperature must remain positive and finite in the logits dtype, and
    division by it must leave finite logits. Checking the latter reads tensor
    values and may synchronize the device. The default is the original pathway
    router's soft assignment, with unchanged arithmetic and no random draws.
    Hard training uses ``discrete + soft - soft.detach()``, the expression of
    the reduced models it replays; a selected entry can then differ from one in
    its last bit (``discrete + (soft - soft.detach())`` would be exact).
    Evaluation uses exact one-hot entries. This is not Top-K contact selection
    within a row.
    """
    _check_operand(temperature, logits, "temperature", positive=True)
    scaled_logits = logits / temperature
    if not bool(torch.isfinite(scaled_logits).all()):
        raise ValueError("scaled routing logits must be finite before softmax")
    soft = torch.softmax(scaled_logits, dim=dim)
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

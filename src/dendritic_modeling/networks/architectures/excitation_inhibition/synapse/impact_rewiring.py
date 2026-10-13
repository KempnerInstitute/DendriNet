"""Cellwise replacement of low-impact indexed contacts, separated by sign."""

import math

import torch

from dendritic_modeling.config.impact_rewiring import ImpactRewiringConfig
from dendritic_modeling.networks.utils.sparse_maintenance import maintenance_applies

from .indexed_release import IndexedReleaseLinear


class CellwiseImpactRewiring:
    """Maintain groups of same-sign synapse banks for a population of cells.

    Each group is (n_cells, banks); rows in each bank must be contiguous per
    cell, as in native DendriNet branch ordering. E and I belong in separate
    groups. Sources are sampled without duplicates within a segment, respecting
    allowed/forbidden indices and the configured regrowth policies. Call after
    optimizer.step(); step-stage impact metrics are accumulated on every call.
    """

    def __init__(self, groups, config: ImpactRewiringConfig):
        self.groups = [(n, tuple(banks)) for n, banks in groups if banks]
        for n, banks in self.groups:
            if n < 1 or any(
                not isinstance(b, IndexedReleaseLinear) or b.out_features % n
                for b in banks
            ):
                raise ValueError(
                    "groups require indexed_release banks with whole rows per cell"
                )
        banks = [bank for _, group in self.groups for bank in group]
        if config.regrowth_sources == "active" and any(
            bank.source_activity is None for bank in banks
        ):
            raise ValueError(
                "regrowth_sources='active' requires release_track_source_activity"
            )
        if config.regrowth_weight == "initial" and any(
            bank.initial_log_mean is None or bank.initial_log_sd is None
            for bank in banks
        ):
            raise ValueError(
                "regrowth_weight='initial' requires banks initialized with a "
                "recorded log-normal law (dbl_init_method='fan_in_lognormal')"
            )
        self.config = config
        self.steps = 0
        self.generator = torch.Generator().manual_seed(config.seed)

    def state_dict(self):
        return {"steps": self.steps, "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        self.steps = int(state["steps"])
        self.generator.set_state(state["generator"].cpu())

    @torch.no_grad()
    def step(self, optimizer):
        # Frozen banks (requires_grad=False, or not owned by the stepping
        # optimizer inside CustomWeightDecayOptimizer) neither age nor rewire.
        for _, banks in self.groups:
            for bank in banks:
                if maintenance_applies(bank.pre_w):
                    bank.accumulate_step_impact()
        self.steps += 1
        cfg = self.config
        if self.steps % cfg.frequency or (
            cfg.until_step is not None and self.steps >= cfg.until_step
        ):
            return 0
        replaced = 0
        for n_cells, banks in self.groups:
            maintained = [maintenance_applies(b.pre_w) for b in banks]
            if not any(maintained):
                continue
            counts = [b.pre_w.numel() // n_cells for b in banks]
            impacts = torch.cat(
                [b.impact.detach().cpu().reshape(n_cells, -1) for b in banks], 1
            )
            ages = torch.cat(
                [b.contact_age.detach().cpu().reshape(n_cells, -1) for b in banks], 1
            )
            values = torch.cat(
                [b.sparse_weight().detach().cpu().reshape(n_cells, -1) for b in banks],
                1,
            )
            eligible = (impacts < cfg.threshold * impacts.mean(1, keepdim=True)) & (
                ages >= cfg.maturity_steps
            )
            if not all(maintained):
                eligible &= torch.cat(
                    [
                        torch.full((n_cells, count), flag, dtype=torch.bool)
                        for flag, count in zip(maintained, counts)
                    ],
                    1,
                )
            selected = torch.zeros_like(eligible)
            for cell in range(n_cells):
                candidates = eligible[cell].nonzero().flatten()
                limit = math.floor(len(candidates) * cfg.eligible_fraction)
                candidates = candidates[
                    torch.randperm(len(candidates), generator=self.generator)
                ]
                used = {}
                accepted = 0
                for index in candidates.tolist():
                    if accepted >= limit:
                        break
                    local = index
                    bank_index = 0
                    for count in counts:
                        if local < count:
                            break
                        local -= count
                        bank_index += 1
                    bank = banks[bank_index]
                    segment = (bank_index, local // bank.K)
                    cap = math.floor(bank.K * cfg.segment_fraction)
                    if used.get(segment, 0) >= cap:
                        continue
                    selected[cell, index] = True
                    used[segment] = used.get(segment, 0) + 1
                    accepted += 1
            offset = 0
            median = values.median(1).values
            for bank, count in zip(banks, counts):
                mask = selected[:, offset : offset + count].reshape_as(bank.pre_w)
                offset += count
                indices = bank.connection_indices.detach().cpu().clone()
                allowed_mask = bank._allowed_connection_mask.detach().cpu()
                forbidden = bank._forbidden_input_index_per_output.detach().cpu()
                active_sources = None
                if cfg.regrowth_sources == "active":
                    active_sources = (
                        bank.source_activity.detach().cpu() > cfg.min_source_activity
                    )
                changed = torch.zeros_like(mask)
                for row in mask.any(1).nonzero().flatten().tolist():
                    allowed = (
                        torch.ones(bank.in_features, dtype=torch.bool)
                        if not allowed_mask.numel()
                        else allowed_mask[row].clone()
                    )
                    allowed[indices[row].long()] = False
                    if forbidden.numel() and forbidden[row] >= 0:
                        allowed[int(forbidden[row])] = False
                    if active_sources is not None:
                        allowed &= active_sources
                    sources = allowed.nonzero().flatten()
                    sources = sources[
                        torch.randperm(len(sources), generator=self.generator)
                    ]
                    slots = mask[row].nonzero().flatten()[: len(sources)]
                    indices[row, slots] = sources[: len(slots)].to(indices.dtype)
                    changed[row, slots] = True
                device_mask = changed.to(bank.pre_w.device)
                if not changed.any():
                    continue
                bank.connection_indices.copy_(indices)
                if cfg.regrowth_weight == "cell_median":
                    new_values = (
                        median[:, None]
                        .expand(n_cells, count)
                        .reshape_as(bank.pre_w)
                        .to(bank.pre_w.device)
                    )[device_mask]
                else:
                    draws = torch.randn(int(changed.sum()), generator=self.generator)
                    new_values = (
                        (bank.initial_log_mean + bank.initial_log_sd * draws)
                        .exp()
                        .to(device=bank.pre_w.device, dtype=bank.pre_w.dtype)
                    )
                bank.pre_w[device_mask] = bank.inverse_weight(new_values)
                bank.impact[device_mask] = 0
                bank.contact_age[device_mask] = 0
                # Reset per-contact optimizer moments, not the scalar step.
                for value in optimizer.state.get(bank.pre_w, {}).values():
                    if torch.is_tensor(value) and value.shape == bank.pre_w.shape:
                        value[device_mask] = 0
                replaced += int(changed.sum())
        return replaced


def population_rewiring_groups(core):
    """(n_cells, banks) groups of a population network, one per population and sign.

    A group holds a population's indexed_release banks of one sign (excitatory
    or inhibitory) at every dendritic level, distal to soma. Groups follow
    layer and population order, which fixes the rewiring random stream.
    """
    layers = getattr(core, "layers", None)
    if layers is None or not all(hasattr(layer, "populations") for layer in layers):
        raise TypeError("impact rewiring groups require a population_network core")
    groups = []
    for layer in layers:
        for population in layer.populations.values():
            for pathway in ("branch_excitation", "branch_inhibition"):
                banks = [
                    getattr(branch, pathway)
                    for branch in population.branch_layers
                    if isinstance(getattr(branch, pathway, None), IndexedReleaseLinear)
                ]
                if banks:
                    groups.append((population.pop_config.n_neurons, banks))
    return groups


def build_population_impact_rewiring(
    core, options, *, steps_per_epoch, epochs, seed
) -> CellwiseImpactRewiring:
    """Impact-rewiring controller from a training ``impact_rewiring`` mapping.

    Keys are ImpactRewiringConfig fields except ``until_step`` and ``seed``,
    plus ``freeze_last_epochs``: wiring stops changing for the final epochs
    while weights continue to train. ``seed`` drives replacement sampling.
    """
    options = dict(options)
    freeze = int(options.pop("freeze_last_epochs", 0))
    for key in ("until_step", "seed"):
        if key in options:
            raise ValueError(
                f"impact_rewiring.{key} is derived; use freeze_last_epochs or "
                "the experiment topology seed"
            )
    if freeze < 0 or freeze > epochs:
        raise ValueError("freeze_last_epochs must be between 0 and the epoch count")
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    config = ImpactRewiringConfig(
        **options,
        seed=int(seed),
        until_step=max(0, int(epochs) - freeze) * int(steps_per_epoch),
    )
    groups = population_rewiring_groups(core)
    if not groups:
        raise ValueError("impact rewiring requires indexed_release synapses")
    return CellwiseImpactRewiring(groups, config)

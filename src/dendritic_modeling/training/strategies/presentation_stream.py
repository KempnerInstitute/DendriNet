"""Training from a stream of freshly sampled presentations.

Each epoch is a fixed number of presentations drawn with replacement from a
task stream rather than one pass over a dataset; corruption and context are
sampled per presentation on the training device. The training set must expose
``stream.sample(count, index_generator=..., noise_generator=...)`` and
``stream.to(device)``, as ``datasets.contextual_remapping`` does.

Per step: learning-rate factor, forward, cross-entropy, backward, optional
value clipping, then the optimizer step. The step goes through
``CustomWeightDecayOptimizer``, which applies maintenance, rewiring and topology
controllers such as impact rewiring. ``stop_after_epochs`` ends a run early
without changing the schedule. After each epoch every validation variant
is evaluated (e.g. noisy and clean). The best epoch is selected on the
``selection_variant`` ("noisy" when present, else the only validation set) by
accuracy with loss as tie-break, or by loss.

Random streams: index and corruption draws use generators seeded with
``seed + sample_seed_offset`` and ``seed + noise_seed_offset`` on the training
device. The device's global generator is reseeded with ``seed`` before the
first step, so stochastic layers (synaptic release) do not depend on setup.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time

import torch
from torch.nn import functional as F

from dendritic_modeling.training.clipping import clip_gradients
from dendritic_modeling.training.loss.functions import CrossEntropyLoss
from dendritic_modeling.training.strategies.standard import LR_SCHEDULES, Trainer

logger = logging.getLogger(__name__)

SELECTION_METRICS = ("accuracy", "loss")


class PresentationStreamTrainer(Trainer):

    # The learning rate follows a per-step schedule (see _lr_factor).
    applies_own_lr_schedule = True
    # Options its own step loop does not apply (see STEP_OPTIONS).
    unapplied_options = Trainer.unapplied_options | frozenset(
        {
            "use_amp",
            "enable_nan_checking",
            "profile_dataloader",
        }
    )

    def __init__(self, optimizer, **kwargs):
        options = dict(kwargs.pop("presentation_stream_config", None) or {})
        super().__init__(optimizer, **kwargs)
        self.presentations_per_epoch = int(
            options.pop("presentations_per_epoch", 65536)
        )
        self.lr_terminal_multiplier = float(options.pop("lr_terminal_multiplier", 1.0))
        self.sample_seed_offset = int(options.pop("sample_seed_offset", 4100))
        self.noise_seed_offset = int(options.pop("noise_seed_offset", 9100))
        self.selection_metric = str(options.pop("selection_metric", "accuracy"))
        self.selection_variant = options.pop("selection_variant", None)
        self.evaluation_batch_size = int(options.pop("evaluation_batch_size", 512))
        self.require_finite = bool(options.pop("require_finite", True))
        self.metrics_file = str(options.pop("metrics_file", "metrics.jsonl"))
        self.stop_after_epochs = int(options.pop("stop_after_epochs", 0))
        if options:
            raise ValueError(f"unknown presentation_stream options: {sorted(options)}")
        self.lr_schedule = str(getattr(self, "lr_schedule", "none")).lower()
        self.lr_warmup_epochs = float(getattr(self, "lr_warmup_epochs", 0) or 0)
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}")
        if self.lr_warmup_epochs < 0:
            raise ValueError("lr_warmup_epochs must be >= 0")
        if self.lr_warmup_epochs > 0 and self.lr_schedule != "cosine":
            raise ValueError(
                "lr_warmup_epochs is part of the cosine schedule; set "
                "lr_schedule: cosine (as in every other trainer)"
            )
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
        if not 0 <= self.stop_after_epochs <= self.epochs:
            raise ValueError("stop_after_epochs must be between 0 and epochs")
        if self.presentations_per_epoch < 1 or self.evaluation_batch_size < 1:
            raise ValueError(
                "presentations_per_epoch and evaluation_batch_size must be positive"
            )
        if not 0 <= self.lr_terminal_multiplier <= 1:
            raise ValueError("lr_terminal_multiplier must be in [0, 1]")
        if self.lr_schedule == "none" and self.lr_terminal_multiplier != 1.0:
            raise ValueError("lr_terminal_multiplier requires lr_schedule='cosine'")
        if self.lr_schedule == "cosine" and self.lr_terminal_multiplier == 1.0:
            raise ValueError(
                "lr_schedule='cosine' decays to lr_terminal_multiplier times the "
                "initial rate; at the default 1.0 the rate never changes. Set "
                "presentation_stream.lr_terminal_multiplier below 1"
            )
        if self.enable_amp or getattr(self, "use_amp", False):
            raise ValueError("presentation_stream does not support use_amp")
        if getattr(self, "recalibrate_reactivation_every", 0):
            raise ValueError(
                "presentation_stream does not support recalibrate_reactivation_every; "
                "stream calibration requires a separate sampling protocol"
            )
        if not isinstance(self.loss_function, CrossEntropyLoss):
            raise ValueError(
                "presentation_stream trains classification with cross-entropy "
                "(loss_function='ce')"
            )
        self.logger_info_prefix = "[PresentationStream] "
        self.filename_prefix = "stream_"

    # ------------------------------------------------------------------
    def _lr_factor(self, step: int, total_steps: int, steps_per_epoch: int) -> float:
        """Learning-rate factor of optimizer step ``step`` (counted from 0).

        ``cosine`` falls from 1 at the first step to ``lr_terminal_multiplier``
        at the last step of the full schedule; during the first
        ``lr_warmup_epochs`` (fractions allowed) it is multiplied by the
        linear ramp ``(step + 1) / warmup_steps``.
        """
        factor = 1.0
        if self.lr_schedule == "cosine":
            progress = step / max(1, total_steps - 1)
            m = self.lr_terminal_multiplier
            factor = m + (1 - m) * 0.5 * (1 + math.cos(math.pi * progress))
        warm_steps = self.lr_warmup_epochs * steps_per_epoch
        if warm_steps > 0:
            factor *= min(1.0, (step + 1) / warm_steps)
        return factor

    @staticmethod
    def _raise_if_not_finite(total, model, epoch: int, step: int) -> None:
        """Raise for a non-finite loss, else for a non-finite gradient."""
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        grads_finite = (
            torch.stack([torch.isfinite(grad).all() for grad in grads]).all()
            if grads
            else torch.ones((), dtype=torch.bool, device=total.device)
        )
        loss_finite, grads_finite = (
            torch.stack([torch.isfinite(total.detach()), grads_finite]).cpu().tolist()
        )
        if not loss_finite:
            raise RuntimeError(f"non-finite loss at epoch {epoch + 1}, step {step}")
        if not grads_finite:
            raise RuntimeError(f"non-finite gradient at epoch {epoch + 1}, step {step}")

    @staticmethod
    def _evaluation_tensors(dataset):
        """{variant: (inputs, targets)} of a fixed evaluation set."""
        variants = getattr(dataset, "variants", None) or {"validation": dataset}
        tensors = {}
        for name, data in variants.items():
            if hasattr(data, "tensors"):
                inputs, targets = data.tensors[:2]
            else:
                pairs = [data[i] for i in range(len(data))]
                inputs = torch.stack([p[0] for p in pairs])
                targets = torch.as_tensor([int(p[1]) for p in pairs])
            tensors[name] = (inputs, targets)
        return tensors

    @torch.no_grad()
    def _evaluate(self, model, inputs, targets) -> dict:
        model.eval()
        loss, correct = 0.0, 0
        for start in range(0, len(targets), self.evaluation_batch_size):
            logits = model(inputs[start : start + self.evaluation_batch_size])
            target = targets[start : start + self.evaluation_batch_size]
            loss += F.cross_entropy(logits, target, reduction="sum").item()
            correct += int(logits.argmax(-1).eq(target).sum())
        return {"loss": loss / len(targets), "accuracy": correct / len(targets)}

    def _score(self, metrics: dict) -> tuple:
        if self.selection_metric == "accuracy":
            return (metrics["accuracy"], -metrics["loss"])
        return (-metrics["loss"],)

    # ------------------------------------------------------------------
    def train(self, model, train_data, valid_data) -> dict:
        if self._is_distributed:
            raise RuntimeError(
                "presentation_stream does not support distributed training"
            )
        stream = getattr(train_data, "stream", None)
        if stream is None or not callable(getattr(stream, "sample", None)):
            raise TypeError(
                "presentation_stream requires a training set with a sampling stream "
                "(for example data.dataset_name: contextual_remapping)"
            )
        self._initialize_attributes(model, self.epochs)
        device = torch.device(self.device)
        stream = stream.to(device)
        evaluation = {
            name: (inputs.to(device), targets.to(device))
            for name, (inputs, targets) in self._evaluation_tensors(valid_data).items()
        }
        if self.selection_variant is None:  # "noisy" when present, else the only set
            self.selection_variant = (
                "noisy"
                if "noisy" in evaluation
                else (next(iter(evaluation)) if len(evaluation) == 1 else None)
            )
        if self.selection_variant not in evaluation:
            raise ValueError(
                f"selection_variant {self.selection_variant!r} is not one of {sorted(evaluation)}"
            )
        steps = math.ceil(self.presentations_per_epoch / self.batch_size)
        total_steps = self.epochs * steps
        base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        else:
            torch.manual_seed(self.seed)
        index_generator = torch.Generator(device=device).manual_seed(
            self.seed + self.sample_seed_offset
        )
        noise_generator = torch.Generator(device=device).manual_seed(
            self.seed + self.noise_seed_offset
        )
        has_regularization = self.regularization_manager.has_regularization()
        metrics_path = (
            os.path.join(self.save_path, self.metrics_file)
            if self.save_path and self._is_main_process
            else None
        )
        if metrics_path and os.path.exists(metrics_path):
            raise FileExistsError(f"{metrics_path} exists; use a new run directory")
        best_score = None
        history, train_losses, valid_losses = [], [], []
        train_losses_base, train_losses_reg = [], []
        for epoch in range(self.epochs):
            self.epoch_counter += 1
            started = time.monotonic()
            model.train()
            # Sums stay on the device and are read once per epoch; float64
            # with a separate multiply and add reproduces the host sums.
            loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            reg_sum = torch.zeros((), dtype=torch.float64, device=device)
            correct = torch.zeros((), dtype=torch.int64, device=device)
            seen = 0
            replaced_before = int(getattr(self.optimizer, "topology_replacements", 0))
            for step in range(steps):
                count = min(self.batch_size, self.presentations_per_epoch - seen)
                inputs, targets = stream.sample(
                    count,
                    index_generator=index_generator,
                    noise_generator=noise_generator,
                )
                factor = self._lr_factor(epoch * steps + step, total_steps, steps)
                for group, base in zip(self.optimizer.param_groups, base_lrs):
                    group["lr"] = base * factor
                self.optimizer.zero_grad()
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
                regularization = None
                if has_regularization:
                    regularization = (
                        self.regularization_manager.compute_regularization_loss(
                            model,
                            inputs,
                            targets,
                            param_groups=self.optimizer.param_groups,
                        )
                    )
                total = loss if regularization is None else loss + regularization
                total.backward()
                if self.require_finite:
                    # One host read per step, before the optimizer step.
                    self._raise_if_not_finite(total, model, epoch, step)
                clip_gradients(
                    model.parameters(), self.grad_clip_value, self.grad_clip_mode
                )
                self.optimizer.step()
                loss_sum += loss.detach().double() * count
                if regularization is not None:
                    reg_sum += regularization.detach().double() * count
                correct += logits.argmax(-1).eq(targets).sum()
                seen += count
            loss_sum, reg_sum = float(loss_sum), float(reg_sum)
            correct = int(correct)
            results = {
                name: self._evaluate(model, *data) for name, data in evaluation.items()
            }
            row = {
                "epoch": self.epoch_counter,
                "train_loss": loss_sum / seen,
                "train_loss_reg": reg_sum / seen,
                "train_loss_total": (loss_sum + reg_sum) / seen,
                "train_accuracy": correct / seen,
                "validation": results,
                "replaced": int(getattr(self.optimizer, "topology_replacements", 0))
                - replaced_before,
                "learning_rate_factor": factor,
                "seconds": time.monotonic() - started,
            }
            history.append(row)
            train_losses.append(row["train_loss_total"])
            train_losses_base.append(row["train_loss"])
            train_losses_reg.append(row["train_loss_reg"])
            valid_losses.append(results[self.selection_variant]["loss"])
            if metrics_path:
                with open(metrics_path, "a") as stream_file:
                    stream_file.write(json.dumps(row) + "\n")
            score = self._score(results[self.selection_variant])
            if best_score is None or score > best_score:
                best_score = score
                self.best_epoch = self.epoch_counter
                self.best_loss = results[self.selection_variant]["loss"]
                self.best_state_dict = self._snapshot_state(model)
                self.best_state_is_initial = False
                self.patience_counter = 0
                if metrics_path and self.save_best_checkpoint == "every_improvement":
                    self._write_best_checkpoint()
            else:
                self.patience_counter += 1
            self._save_checkpoint(model)
            self._analyze()
            if not self.suppress_prints and self.epoch_counter % self.print_every == 0:
                logger.info(
                    "%sepoch=%d/%d train_loss=%.4f %s",
                    self.logger_info_prefix,
                    self.epoch_counter,
                    self.epochs,
                    row["train_loss"],
                    " ".join(
                        f"{name}_acc={value['accuracy']:.4f}"
                        for name, value in results.items()
                    ),
                )
            if self._check_early_stopping():
                break
            if self.stop_after_epochs and self.epoch_counter >= self.stop_after_epochs:
                break  # partial run of the full schedule (e.g. a replay check)
        self._restore_best_state(model, valid_losses)
        return {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "best_state_dict": self.best_state_dict,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "train_losses_base": train_losses_base,
            "train_losses_reg": train_losses_reg,
            "history": history,
            "steps_per_epoch": steps,
        }

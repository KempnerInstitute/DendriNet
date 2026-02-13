"""
Representation probes: per-layer activations, linear probe accuracy, CKA.
Run after baseline and single-swap ablations are stable.
"""

import json
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


def collect_layer_activations(
    model: nn.Module,
    encoder_attr: str,
    layer_names: List[str],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Register forward hooks on named layers, return layer_name -> (N, D) activations."""
    encoder = getattr(model, encoder_attr, None) or getattr(model, "encoder_network", None)
    if encoder is None:
        return {}
    encoder = getattr(encoder, "encoder", encoder)
    activations: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}
    handles = []

    def make_hook(name: str):
        def hook(_m, _in, out):
            activations[name].append(out.detach().cpu())
        return hook

    for name, mod in encoder.named_modules():
        if name in layer_names:
            handles.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            _ = model(x.to(device))

    for h in handles:
        h.remove()
    return {k: torch.cat(v, dim=0) for k, v in activations.items() if v}


def linear_probe_accuracy(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_frac: float = 0.8,
    max_iter: int = 1000,
) -> float:
    """Train linear classifier on features, return test accuracy."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return -1.0
    N = features.shape[0]
    X = features.numpy()
    y = labels.numpy()
    perm = torch.randperm(N).numpy()
    n_train = int(N * train_frac)
    X_train, X_test = X[perm[:n_train]], X[perm[n_train:]]
    y_train, y_test = y[perm[:n_train]], y[perm[n_train:]]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Centered Kernel Alignment (minimal impl in Adi)."""
    X = (X - X.mean(dim=0)).double()
    Y = (Y - Y.mean(dim=0)).double()
    n = X.shape[0]
    if n < 2:
        return 0.0
    XtY = X.t() @ Y
    denom = ((X.t() @ X).norm() * (Y.t() @ Y).norm() + 1e-12).item()
    return (XtY ** 2).sum().item() / denom


def run_probes(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: str,
    layer_names: Optional[List[str]] = None,
    max_batches: Optional[int] = 50,
) -> Dict[str, Any]:
    """Collect activations, linear probe per layer, CKA between layers; save JSON."""
    layer_names = layer_names or ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7"]
    activations = collect_layer_activations(
        model, "encoder_network", layer_names, dataloader, device, max_batches
    )
    if not activations:
        return {"error": "no activations"}

    labels_list = []
    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        if isinstance(batch, (list, tuple)) and len(batch) > 1:
            labels_list.append(batch[1])
    labels = torch.cat(labels_list, dim=0).cpu() if labels_list else None

    results: Dict[str, Any] = {}
    for name, act in activations.items():
        flat = act.reshape(act.shape[0], -1)
        results[name] = {"shape": list(act.shape)}
        if labels is not None and flat.shape[0] == labels.shape[0]:
            results[name]["linear_probe_accuracy"] = linear_probe_accuracy(flat, labels)

    names = [n for n in layer_names if n in activations]
    for i in range(len(names) - 1):
        a = activations[names[i]].reshape(activations[names[i]].shape[0], -1)
        b = activations[names[i + 1]].reshape(activations[names[i + 1]].shape[0], -1)
        if a.shape[0] == b.shape[0]:
            results["cka_%s_%s" % (names[i], names[i + 1])] = cka(a, b)

    out = {}
    for k, v in results.items():
        if isinstance(v, dict):
            out[k] = {kk: (vv if not isinstance(vv, torch.Tensor) else vv.tolist()) for kk, vv in v.items()}
        elif isinstance(v, (int, float, str)):
            out[k] = v
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    return results

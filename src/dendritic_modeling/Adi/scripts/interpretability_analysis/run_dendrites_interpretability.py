import sys
sys.path.insert(0, '/n/home13/adithyavm/dendritic-modeling/src')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader

INTERP   = Path('/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/adivm/dendritic-modeling/outputs/interpretability_results')
OUTPUTS  = Path('/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/adivm/dendritic-modeling/outputs')
NB_DIR   = Path('/n/home13/adithyavm/dendritic-modeling/src/dendritic_modeling/Adi/scripts/interpretability_analysis')
FIG_DIR  = NB_DIR / 'figures_dendrites'
FIG_DIR.mkdir(exist_ok=True)

IMAGENET  = '/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k'
N_CLASSES = 1000
device    = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
print(f'figures → {FIG_DIR}')

CONFIG_PATHS = {
    '2x2':   '/n/home13/adithyavm/dendritic-modeling/configs/vision/alexnet_k1_ne4096_noproj_ddp.yaml',
    '3x3x3': '/n/home13/adithyavm/dendritic-modeling/configs/vision/alexnet_k1_ne4096_3x3x3_noproj_ddp.yaml',
}
RUN_DIRS = {
    '2x2':   sorted(OUTPUTS.glob('alexnet_k1_ne4096_noproj_ddp_*'))[-1],
    '3x3x3': sorted(OUTPUTS.glob('alexnet_k1_ne4096_3x3x3_noproj_ddp_*'))[-1],
}


def class_mean_and_sparseness(acts, labels, n_classes=1000):
    n_units = acts.shape[1]
    pcm = np.zeros((n_units, n_classes), dtype=np.float32)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            pcm[:, c] = acts[mask].mean(axis=0)
    r_pos = np.maximum(pcm, 0)
    n = n_classes
    num   = (r_pos.sum(axis=1) / n) ** 2
    denom = (r_pos ** 2).sum(axis=1) / n
    S = np.where(denom > 1e-10, (1 - num / denom) / (1 - 1/n), 0.0)
    return pcm, S.astype(np.float32)


for MORPH in ['2x2', '3x3x3']:
    print(f'\n{"="*60}')
    print(f'MORPH = {MORPH}')
    print(f'{"="*60}')

    from dendritic_modeling.config import load_config
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

    config   = load_config(CONFIG_PATHS[MORPH])
    model, _ = initialize_model(config.model)
    ckpt     = torch.load(RUN_DIRS[MORPH] / 'final_model.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval().to(device)
    print(f'Model loaded from {RUN_DIRS[MORPH]}')

    branch_layers = []
    for ei_layer in model.core_network.layers:
        if hasattr(ei_layer, 'excitatory_cells'):
            branch_layers.extend(ei_layer.excitatory_cells.branch_layers)

    n_layers = len(branch_layers)
    print(f'Excitatory branch layers (including soma): {n_layers}')
    for i, bl in enumerate(branch_layers):
        lbl = 'SOMA' if i == n_layers - 1 else f'branch layer {i}'
        print(f'  [{i}] {lbl}: n_branches={bl.n_branches}')

    # ── collect activations via forward hooks ─────────────────────────────────
    n_dendritic_layers = n_layers - 1
    branch_acts = [[] for _ in range(n_dendritic_layers)]
    soma_acts   = []
    all_labels  = []

    handles = []
    for i in range(n_dendritic_layers):
        def make_hook(layer_idx):
            def hook(module, inp, out):
                branch_acts[layer_idx].append(out.detach().cpu().float())
            return hook
        handles.append(branch_layers[i].register_forward_hook(make_hook(i)))

    def soma_hook(module, inp, out):
        if isinstance(out, tuple): out = out[0]
        soma_acts.append(out.detach().cpu().float())
    handles.append(model.core_network.register_forward_hook(soma_hook))

    from dendritic_modeling.datasets.standard_datasets import load_imagenet_as_datasets
    _, _, val_ds = load_imagenet_as_datasets(
        train_valid_split=0.8, flatten=False, normalize=True, task_data_path=IMAGENET)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    print('Running val set...')
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(val_loader):
            model(imgs.to(device))
            all_labels.append(targets)
            if (batch_idx + 1) % 50 == 0:
                print(f'  {(batch_idx+1) * 128} images...')

    for h in handles:
        h.remove()

    labels_np  = torch.cat(all_labels).numpy()
    soma_np    = torch.cat(soma_acts, dim=0).numpy()
    branch_nps = [torch.cat(b, dim=0).numpy() for b in branch_acts]
    print(f'Soma: {soma_np.shape}')
    for i, b in enumerate(branch_nps):
        print(f'Branch layer {i}: {b.shape}')

    # ── sparseness ────────────────────────────────────────────────────────────
    print('Computing sparseness...')
    soma_pcm, soma_S = class_mean_and_sparseness(soma_np, labels_np)
    print(f'  soma: S mean={soma_S.mean():.4f}  dead={(soma_S==0).sum()}')

    branch_pcms, branch_Ss = [], []
    for i, b in enumerate(branch_nps):
        pcm, S = class_mean_and_sparseness(b, labels_np)
        branch_pcms.append(pcm)
        branch_Ss.append(S)
        print(f'  layer {i}: S mean={S.mean():.4f}  dead={(S==0).sum()}')

    all_layers = branch_Ss + [soma_S]
    layer_labels = [f'layer {i} (tips)' if i == 0 else
                    (f'layer {i}' if i < len(branch_Ss) else 'soma')
                    for i in range(len(all_layers))]

    # ── Figure 1: sparseness distribution tips → soma ─────────────────────────
    fig, axes = plt.subplots(1, len(all_layers), figsize=(5*len(all_layers), 4), sharey=False)
    if len(all_layers) == 1: axes = [axes]
    cmap = plt.cm.viridis
    colors = [cmap(i/(len(all_layers)-1)) for i in range(len(all_layers))]
    for ax, S, lbl, c in zip(axes, all_layers, layer_labels, colors):
        ax.hist(S, bins=80, color=c, edgecolor='none', alpha=0.85)
        ax.axvline(S.mean(), color='#333', lw=1.2, ls='--', label=f'mean={S.mean():.4f}')
        ax.set_xlabel('lifetime sparseness S', fontsize=10)
        ax.set_ylabel('# units', fontsize=10)
        ax.set_title(f'{lbl}\nn={len(S)}', fontsize=10)
        ax.legend(fontsize=8)
        ax.spines[['top','right']].set_visible(False)
    plt.suptitle(f'{MORPH}: lifetime sparseness from tips to soma', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'fig_branch_sparseness_{MORPH}.pdf', bbox_inches='tight')
    plt.close()
    print(f'Saved fig_branch_sparseness_{MORPH}.pdf')

    # ── Figure 2: raster map ──────────────────────────────────────────────────
    all_pcms = branch_pcms + [soma_pcm]
    all_labels_raster = [f'layer {i}' for i in range(len(branch_pcms))] + ['soma']
    fig, axes = plt.subplots(1, len(all_pcms), figsize=(6*len(all_pcms), 5))
    if len(all_pcms) == 1: axes = [axes]
    for ax, pcm, lbl in zip(axes, all_pcms, all_labels_raster):
        order      = np.argsort(pcm.argmax(axis=1))
        subsample  = max(1, len(order) // 1024)
        pcm_sorted = pcm[order][::subsample]
        vmax = np.percentile(pcm_sorted, 99)
        im = ax.imshow(pcm_sorted, aspect='auto', cmap='inferno',
                       vmin=0, vmax=vmax, interpolation='nearest')
        ax.set_xlabel('ImageNet class', fontsize=9)
        ax.set_ylabel('Unit (sorted by pref. class)', fontsize=9)
        ax.set_title(f'{lbl}  n={pcm.shape[0]}', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, label='mean act')
    plt.suptitle(f'{MORPH}: per-class mean activation raster — tips to soma', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'fig_branch_raster_{MORPH}.pdf', bbox_inches='tight')
    plt.close()
    print(f'Saved fig_branch_raster_{MORPH}.pdf')

    # ── Figure 3: mean S across layers ────────────────────────────────────────
    means   = [S.mean() for S in all_layers]
    medians = [np.median(S[S > 0]) if (S > 0).sum() > 0 else 0 for S in all_layers]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(means))
    ax.plot(x, means,   'o-', color='#2a78d6', lw=2, ms=7, label='mean S (all units)')
    ax.plot(x, medians, 's--', color='#e07b39', lw=1.5, ms=6, label='median S (active only)')
    ax.set_xticks(x); ax.set_xticklabels(layer_labels, rotation=20, ha='right')
    ax.set_ylabel('lifetime sparseness S', fontsize=11)
    ax.set_title(f'{MORPH}: does selectivity increase from tips to soma?', fontsize=12)
    ax.legend(fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'fig_S_by_layer_{MORPH}.pdf', bbox_inches='tight')
    plt.close()
    print(f'Saved fig_S_by_layer_{MORPH}.pdf')

    # ── Figure 4: branch S vs soma S per neuron ───────────────────────────────
    N_E = soma_np.shape[1]
    fig, axes = plt.subplots(1, len(branch_Ss), figsize=(5*len(branch_Ss), 5))
    if len(branch_Ss) == 1: axes = [axes]
    for ax, (S_branch, lbl) in zip(axes, zip(branch_Ss, layer_labels)):
        n_branches_per_neuron = len(S_branch) // N_E
        if len(S_branch) % N_E != 0:
            print(f'Warning: {lbl} n_branches={len(S_branch)} not divisible by N_E={N_E}')
            ax.set_visible(False)
            continue
        S_branch_per_neuron = S_branch.reshape(N_E, n_branches_per_neuron).mean(axis=1)
        ax.scatter(S_branch_per_neuron, soma_S, s=2, alpha=0.3, rasterized=True, color='#8e44ad')
        lim = [0, max(S_branch_per_neuron.max(), soma_S.max()) * 1.05]
        ax.plot(lim, lim, color='#333', lw=1, ls='--', alpha=0.5)
        r = np.corrcoef(S_branch_per_neuron, soma_S)[0,1]
        ax.set_xlabel(f'mean branch S ({lbl})', fontsize=10)
        ax.set_ylabel('soma S', fontsize=10)
        ax.set_title(f'r={r:.3f}  branches/neuron={n_branches_per_neuron}', fontsize=10)
        ax.spines[['top','right']].set_visible(False)
    plt.suptitle(f'{MORPH}: branch S vs soma S per neuron', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'fig_branch_vs_soma_S_{MORPH}.pdf', bbox_inches='tight')
    plt.close()
    print(f'Saved fig_branch_vs_soma_S_{MORPH}.pdf')

    print(f'\nDone with {MORPH}.')

print(f'\nAll figures saved to {FIG_DIR}')

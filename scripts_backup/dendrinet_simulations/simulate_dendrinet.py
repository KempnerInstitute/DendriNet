#!/usr/bin/env python3
"""
enumerate_correlation_and_group_assignments.py

We create a D-dimensional feature space partitioned into 3 equal groups:
  - 0-info
  - medium-info
  - high-info
Each group has D/3 dimensions. We choose N_E excitatory dims entirely from
one of those groups, and N_I inhibitory dims entirely from one group (possibly
the same or different). => 9 ways (3 choices for E-group x 3 for I-group).

We also consider correlation combos for E–E, I–I, E–I in {-rho, 0, +rho},
i.e. 3x3x3 = 27 combos. For each valid covariance matrix, we generate data,
train a single-branch dendritic model in two modes:
  (a) Shunting: out = E / (E + I + eps)
  (b) Linear: out = E - I
and store final performance (accuracy, MI) plus synergy & redundancy.

Finally, we save the results in a CSV. You can do post-hoc plotting to see
which combos yield the best accuracy or synergy.

"""

import itertools
import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

###############################################################################
# CONFIG
###############################################################################

RESULTS_FOLDER = "./results_corr_group_enumeration"
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# Set up logging
LOG_FILE = os.path.join(RESULTS_FOLDER, "simulation.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Plot settings
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context(
    "paper", rc={"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 11}
)

###############################################################################
# 1. DATA GENERATION UTILITIES
###############################################################################


def generate_label_means(D, C):
    """
    Partition D dims into three groups of size D//3 each:
      - group1: no info => mean=0
      - group2: medium => mean=(c)/(3*C)
      - group3: high  => mean=(c)/C
    Return means[c, d].
    """
    means = np.zeros((C, D), dtype=float)
    dgrp = D // 3
    for c in range(C):
        # medium
        means[c, dgrp : 2 * dgrp] = c / (10.0 * C)
        # high
        means[c, 2 * dgrp :] = c / (5.0 * C)
    return means


def build_block_cov(E_dims, I_dims, cEE, cII, cEI, D):
    """
    Build a (D x D) covariance matrix that's mostly identity except for:
      - E–E block has correlation cEE off-diagonal,
      - I–I block has correlation cII off-diagonal,
      - E–I cross-block has correlation cEI,
    For dims not in E_dims or I_dims, keep them identity and 0 cross-corr
    (since we won't actually use them, but we need a full D x D matrix).
    """

    cov = np.eye(D)  # start with identity
    # fill E–E
    for i in E_dims:
        for j in E_dims:
            if i != j:
                cov[i, j] = cEE
    # fill I–I
    for i in I_dims:
        for j in I_dims:
            if i != j:
                cov[i, j] = cII
    # fill E–I
    for i in E_dims:
        for j in I_dims:
            cov[i, j] = cEI
            cov[j, i] = cEI
    return cov


def generate_data(
    D, C, E_dims, I_dims, cEE, cII, cEI, random_state=0, num_samples=2000
):
    """
    1) Build block covariance matrix for dims in E_dims and I_dims.
    2) Attempt Cholesky => if fails => raise error.
    3) Generate y in [0..C-1], then means => shape (C, D).
       For each sample, pick y[i], add mean[y[i]], add correlated noise.
    """
    rng = np.random.default_rng(random_state)
    cov = build_block_cov(E_dims, I_dims, cEE, cII, cEI, D)
    # test PSD
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Cov matrix not PSD for cEE=%.2f,cII=%.2f,cEI=%.2f" % (cEE, cII, cEI)
        )

    means_mat = generate_label_means(D, C)
    y = rng.integers(low=0, high=C, size=num_samples)
    X_base = np.zeros((num_samples, D))

    for i in range(num_samples):
        c_i = y[i]
        mu_i = means_mat[c_i]
        noise = rng.normal(loc=0, scale=1, size=D)
        X_base[i] = mu_i + noise

    X_corr = X_base @ L.T
    return X_corr, y


###############################################################################
# 2. DENDRITIC MODEL: SHUNTING vs. LINEAR
###############################################################################


def forward_pass(X, w_e, w_i, model_type="shunting", eps=1.0, apply_sigmoid=True):
    """
    E = sum_{d} w_e[d]*X[:,d], I = sum_{d} w_i[d]*X[:,d]
    if shunting => out = E / (E + I + eps)
    if linear => out = E - I

    If apply_sigmoid is True, inputs are passed through a sigmoid function
    before being processed by the synaptic weights.
    """
    # Apply sigmoid normalization to inputs if requested
    if apply_sigmoid:
        X_processed = 1.0 / (1.0 + np.exp(-X))
    else:
        X_processed = X

    E = np.sum(w_e * X_processed, axis=1)
    I = np.sum(w_i * X_processed, axis=1)
    if model_type == "shunting":
        return E / (E + I + eps)
    else:
        return E - I


###############################################################################
# 3. MUTUAL INFORMATION, DISCRETIZE
###############################################################################


def discretize(values, num_bins=10):
    mn, mx = np.min(values), np.max(values)
    if mn == mx:
        return np.zeros_like(values, dtype=int)
    edges = np.linspace(mn, mx, num_bins + 1)
    inds = np.digitize(values, edges) - 1
    inds = np.clip(inds, 0, num_bins - 1)
    return inds


def mutual_information(x, y, num_bins=10):
    x_disc = discretize(x, num_bins=num_bins)
    return mutual_info_score(x_disc, y)


def negative_mi(X, y, w_e, w_i, model_type="shunting", apply_sigmoid=True):
    out = forward_pass(
        X, w_e, w_i, model_type=model_type, eps=1.0, apply_sigmoid=apply_sigmoid
    )
    return -mutual_information(out, y, num_bins=10)


###############################################################################
# 4. FINITE-DIFFERENCE FIT
###############################################################################


def fit_finite_diff(
    X,
    y,
    e_dims,
    i_dims,
    model_type="shunting",
    lr=0.05,
    max_iter=20,
    seed=0,
    apply_sigmoid=True,
):
    """
    We'll have w_e, w_i of length D. e_dims are excitatory => can be >0,
    i_dims => can be >0, the rest =0. We do naive finite difference on e_dims
    and i_dims to maximize MI (minimize negative MI).

    Parameters:
    - X: Input data matrix
    - y: Target labels
    - e_dims: Indices of excitatory dimensions
    - i_dims: Indices of inhibitory dimensions
    - model_type: "shunting" or "linear"
    - lr: Learning rate
    - max_iter: Maximum number of iterations
    - seed: Random seed
    - apply_sigmoid: Whether to apply sigmoid to inputs before processing
    """
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    w_e = np.zeros(D)
    w_i = np.zeros(D)

    # initialize
    w_e[e_dims] = rng.random(len(e_dims))
    w_i[i_dims] = rng.random(len(i_dims))

    delta = 1e-3

    def objective(we, wi):
        return negative_mi(
            X, y, we, wi, model_type=model_type, apply_sigmoid=apply_sigmoid
        )

    # Initial objective value
    initial_obj = objective(w_e, w_i)
    current_obj = initial_obj
    logger.debug(f"Initial negative MI: {initial_obj:.6f}")

    # Track progress
    history = []

    for step in range(max_iter):
        base_obj = objective(w_e, w_i)

        # grad wrt w_e
        grad_e = np.zeros_like(w_e)
        for d in e_dims:
            we_plus = w_e.copy()
            we_plus[d] += delta
            obj_plus = objective(we_plus, w_i)
            we_minus = w_e.copy()
            we_minus[d] -= delta
            obj_minus = objective(we_minus, w_i)
            grad_e[d] = (obj_plus - obj_minus) / (2 * delta)

        # grad wrt w_i
        grad_i = np.zeros_like(w_i)
        for d in i_dims:
            wi_plus = w_i.copy()
            wi_plus[d] += delta
            obj_plus = objective(w_e, wi_plus)
            wi_minus = w_i.copy()
            wi_minus[d] -= delta
            obj_minus = objective(w_e, wi_minus)
            grad_i[d] = (obj_plus - obj_minus) / (2 * delta)

        # Update weights
        w_e -= lr * grad_e
        w_i -= lr * grad_i

        # Check progress
        new_obj = objective(w_e, w_i)
        improvement = current_obj - new_obj
        current_obj = new_obj

        # Log every few steps or if significant improvement
        if step % 5 == 0 or improvement > 0.01:
            logger.debug(
                f"Step {step}: neg MI = {new_obj:.6f}, improvement = {improvement:.6f}"
            )

        # Store history
        history.append({"step": step, "neg_mi": new_obj, "improvement": improvement})

    # Log final improvement
    total_improvement = initial_obj - current_obj
    logger.debug(
        f"Finished training {model_type} model. Total improvement: {total_improvement:.6f}"
    )

    return w_e, w_i


###############################################################################
# 5. SYNERGY & REDUNDANCY
###############################################################################


def synergy_redundancy(x1, x2, y, num_bins=5):
    """
    synergy = I(x1,x2; y) - I(x1;y) - I(x2;y)
    redundancy = I(x1;y) + I(x2;y) - I(x1,x2; y)
    """
    d1 = discretize(x1, num_bins)
    d2 = discretize(x2, num_bins)
    joint = d1 * num_bins + d2

    i1 = mutual_info_score(d1, y)
    i2 = mutual_info_score(d2, y)
    i12 = mutual_info_score(joint, y)
    syn = i12 - i1 - i2
    red = i1 + i2 - i12
    return syn, red


def measure_syn_redundancy(X, w_e, w_i, e_dims, i_dims, y):
    """
    sE[k] = w_e[e_dims[k]] * X[:, e_dims[k]]
    sI[k] = w_i[i_dims[k]] * X[:, i_dims[k]]

    Then E–E, I–I, E–I pairs => synergy, redundancy => average them.
    """
    sE = [w_e[d] * X[:, d] for d in e_dims]
    sI = [w_i[d] * X[:, d] for d in i_dims]

    import itertools

    synergy_ee, synergy_ii, synergy_ei = [], [], []
    red_ee, red_ii, red_ei = [], [], []

    # E–E
    for a, b in itertools.combinations(range(len(sE)), 2):
        syn, rd = synergy_redundancy(sE[a], sE[b], y)
        synergy_ee.append(syn)
        red_ee.append(rd)

    # I–I
    for a, b in itertools.combinations(range(len(sI)), 2):
        syn, rd = synergy_redundancy(sI[a], sI[b], y)
        synergy_ii.append(syn)
        red_ii.append(rd)

    # E–I
    for se in sE:
        for si in sI:
            syn, rd = synergy_redundancy(se, si, y)
            synergy_ei.append(syn)
            red_ei.append(rd)

    def avg_or_zero(lst):
        return np.mean(lst) if len(lst) > 0 else 0.0

    return {
        "syn_ee": avg_or_zero(synergy_ee),
        "syn_ii": avg_or_zero(synergy_ii),
        "syn_ei": avg_or_zero(synergy_ei),
        "red_ee": avg_or_zero(red_ee),
        "red_ii": avg_or_zero(red_ii),
        "red_ei": avg_or_zero(red_ei),
    }


###############################################################################
# 6. FULL ENUMERATION: 27 correlation combos x 9 group assignments x 2 model types
###############################################################################


def run_full_enumeration(
    D=30, C=3, N_E=5, N_I=5, rho=0.3, seeds=[0], max_iter=20, apply_sigmoid=True
):
    """
    We'll assume D is multiple of 3, so each of the 3 info groups has D/3 dims:
      - group0: no-info => dims in [0 : dgrp)
      - group1: medium => dims in [dgrp : 2*dgrp)
      - group2: high => dims in [2*dgrp : 3*dgrp)
    We choose E-group in {0,1,2}, I-group in {0,1,2}, for total 9 combos.
    Then for correlation combos (cEE,cII,cEI) in {-rho,0,rho}^3 => 27 combos.
    We'll do 1 or more seeds.

    For each combo:
      - pick E_dims from the chosen E-group (N_E distinct dims),
        I_dims from chosen I-group (N_I distinct dims),
      - build block cov => if PSD => generate data => fit shunting => fit linear
      - store final results in DataFrame

    Return the DF.
    """
    start_time = time.time()
    logger.info(
        f"Starting full enumeration with D={D}, C={C}, N_E={N_E}, N_I={N_I}, rho={rho}"
    )
    logger.info(f"Apply sigmoid to inputs: {apply_sigmoid}")

    # Check dimensions
    dgrp = D // 3
    if N_E > dgrp or N_I > dgrp:
        logger.error(
            f"Error: Cannot pick N_E={N_E} or N_I={N_I} dimensions from groups of size {dgrp}."
        )
        logger.error(f"Each group has approximately {dgrp} dimensions (D/3 = {D}/3).")
        logger.error(f"Please ensure N_E and N_I are both <= {dgrp}.")
        return pd.DataFrame()  # Return empty dataframe

    rng = np.random.default_rng(9999)

    # group0 => dims [0:dgrp], group1 => [dgrp:2*dgrp], group2 => [2*dgrp:3*dgrp]
    group_ranges = {
        0: np.arange(0, dgrp),
        1: np.arange(dgrp, 2 * dgrp),
        2: np.arange(2 * dgrp, 3 * dgrp),
    }

    corr_values = [-rho, 0.0, +rho]
    all_results = []

    # 9 ways for (E_group, I_group)
    group_pairs = list(itertools.product([0, 1, 2], [0, 1, 2]))
    logger.info(f"Testing {len(group_pairs)} group combinations")

    # 27 ways for (cEE, cII, cEI)
    corr_triples = list(itertools.product(corr_values, corr_values, corr_values))
    logger.info(f"Testing {len(corr_triples)} correlation combinations")

    # Count total valid combinations for progress tracking
    total_valid_combos = 0
    valid_group_pairs = []

    for E_grp, I_grp in group_pairs:
        grpE = group_ranges[E_grp]
        grpI = group_ranges[I_grp]
        if not ((N_E > len(grpE)) or (N_I > len(grpI))):
            valid_group_pairs.append((E_grp, I_grp))
            total_valid_combos += (
                len(corr_triples) * len(seeds) * 2
            )  # *2 for model types

    logger.info(
        f"Found {len(valid_group_pairs)} valid group pairs out of {len(group_pairs)}"
    )
    logger.info(
        f"Maximum possible combinations: {total_valid_combos} (before PSD check)"
    )

    if len(valid_group_pairs) == 0:
        logger.error("No valid group pairs found with the given parameters.")
        return pd.DataFrame()  # Return empty dataframe

    # Track skipped combinations due to PSD failures
    skipped_count = 0
    progress_bar = tqdm(total=total_valid_combos, desc="Processing combinations")

    for E_grp, I_grp in valid_group_pairs:
        # We must pick N_E distinct dims from group E_grp, N_I from group I_grp
        grpE = group_ranges[E_grp]
        grpI = group_ranges[I_grp]

        logger.info(f"Processing E_group={E_grp}, I_group={I_grp}")

        # We'll do a random draw of dims from each group for each seed.
        # (If you want all combinations inside each group, that can explode quickly.)
        # We'll do 1 random assignment per seed for E, 1 random assignment for I.
        for seed in seeds:
            rng2 = np.random.default_rng(seed)
            chosen_e = rng2.choice(grpE, size=N_E, replace=False)
            chosen_i = rng2.choice(grpI, size=N_I, replace=False)

            logger.debug(f"Seed {seed}: Selected E dims {chosen_e}, I dims {chosen_i}")

            # Now the dimension set is chosen_e for excitatory, chosen_i for inhibitory.
            for cEE, cII, cEI in corr_triples:
                # build data
                try:
                    X, y = generate_data(
                        D,
                        C,
                        chosen_e,
                        chosen_i,
                        cEE,
                        cII,
                        cEI,
                        random_state=seed,
                        num_samples=2000,
                    )
                except ValueError:
                    # not PSD => skip
                    skipped_count += 2  # Skip both model types
                    progress_bar.update(2)
                    continue

                for model_type in ["shunting", "linear"]:
                    # Fit
                    logger.debug(
                        f"Fitting {model_type} model with cEE={cEE}, cII={cII}, cEI={cEI}"
                    )
                    w_e, w_i = fit_finite_diff(
                        X,
                        y,
                        e_dims=chosen_e,
                        i_dims=chosen_i,
                        model_type=model_type,
                        lr=0.05,
                        max_iter=max_iter,
                        seed=seed,
                        apply_sigmoid=apply_sigmoid,
                    )
                    # measure final MI
                    out = forward_pass(
                        X, w_e, w_i, model_type=model_type, apply_sigmoid=apply_sigmoid
                    )
                    final_mi = mutual_information(out, y, num_bins=10)

                    # approximate accuracy
                    c_bins = C
                    out_disc = discretize(out, num_bins=c_bins)
                    pred = np.zeros_like(out_disc)
                    for b in range(c_bins):
                        mask = out_disc == b
                        if np.sum(mask) > 0:
                            maj = np.argmax(np.bincount(y[mask], minlength=C))
                            pred[mask] = maj
                    accuracy = np.mean(pred == y)

                    # synergy & redundancy
                    sr = measure_syn_redundancy(X, w_e, w_i, chosen_e, chosen_i, y)

                    row = {
                        "E_group": E_grp,
                        "I_group": I_grp,
                        "cEE": cEE,
                        "cII": cII,
                        "cEI": cEI,
                        "seed": seed,
                        "model_type": model_type,
                        "final_mi": final_mi,
                        "accuracy": accuracy,
                        "apply_sigmoid": apply_sigmoid,
                        **sr,
                    }
                    all_results.append(row)
                    progress_bar.update(1)

    progress_bar.close()
    df = pd.DataFrame(all_results)

    elapsed_time = time.time() - start_time
    logger.info(f"Enumeration complete. Generated {len(df)} combinations.")
    logger.info(
        f"Skipped {skipped_count} combinations due to non-PSD covariance matrices."
    )
    logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds")

    return df


###############################################################################
# 7. VISUALIZATION FUNCTIONS
###############################################################################


def create_heatmap(
    df,
    row_var,
    col_var,
    value_var,
    title,
    filename,
    model_type=None,
    cmap="viridis",
    annot=True,
):
    """
    Create a heatmap of a specific metric across two categorical variables.

    Parameters:
    - df: DataFrame with the data
    - row_var: Variable for rows
    - col_var: Variable for columns
    - value_var: Metric to visualize (e.g., 'accuracy', 'final_mi')
    - title: Plot title
    - filename: Output filename
    - model_type: If provided, filter data to this model type
    - cmap: Colormap to use
    - annot: Whether to annotate cells with values
    """
    plt.figure(figsize=(10, 8))

    # Filter by model type if specified
    plot_df = df.copy()
    if model_type is not None:
        plot_df = plot_df[plot_df["model_type"] == model_type]

    # Create pivot table
    pivot = plot_df.pivot_table(
        index=row_var, columns=col_var, values=value_var, aggfunc="mean"
    )

    # Create heatmap
    ax = sns.heatmap(
        pivot, annot=annot, fmt=".3f", cmap=cmap, cbar_kws={"label": value_var}
    )

    # Add model type to title if specified
    full_title = title
    if model_type is not None:
        full_title += f" ({model_type} model)"

    plt.title(full_title)
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(RESULTS_FOLDER, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved heatmap to {output_path}")


def create_grouped_bar_plot(
    df, x, y, hue, title, filename, palette="Set2", figsize=(12, 8)
):
    """
    Create a grouped bar plot showing a metric across groups.

    Parameters:
    - df: DataFrame with the data
    - x: X-axis variable
    - y: Y-axis metric
    - hue: Grouping variable
    - title: Plot title
    - filename: Output filename
    - palette: Color palette
    - figsize: Figure size
    """
    plt.figure(figsize=figsize)

    ax = sns.barplot(data=df, x=x, y=y, hue=hue, palette=palette)

    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(RESULTS_FOLDER, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved bar plot to {output_path}")


def create_pairplot(
    df, vars_to_plot, hue, title, filename, diag_kind="kde", corner=True, height=2.5
):
    """
    Create a pairplot to visualize relationships between multiple variables.

    Parameters:
    - df: DataFrame with the data
    - vars_to_plot: List of variables to include
    - hue: Variable for coloring
    - title: Plot title
    - filename: Output filename
    - diag_kind: Kind of plot for diagonal
    - corner: Whether to create a corner plot (lower triangle only)
    - height: Height of each facet
    """
    # Create pairplot
    g = sns.pairplot(
        data=df,
        vars=vars_to_plot,
        hue=hue,
        diag_kind=diag_kind,
        corner=corner,
        height=height,
    )

    g.fig.suptitle(title, y=1.02)
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(RESULTS_FOLDER, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved pairplot to {output_path}")


def create_synergy_plots(df, filename_prefix):
    """
    Create specialized plots to visualize synergy patterns.

    Parameters:
    - df: DataFrame with the synergy and redundancy data
    - filename_prefix: Prefix for output filenames
    """
    # Melt the dataframe to get synergy and redundancy in long format
    synergy_cols = ["syn_ee", "syn_ii", "syn_ei"]
    redundancy_cols = ["red_ee", "red_ii", "red_ei"]

    syn_df = df.melt(
        id_vars=[
            "E_group",
            "I_group",
            "cEE",
            "cII",
            "cEI",
            "model_type",
            "accuracy",
            "final_mi",
        ],
        value_vars=synergy_cols,
        var_name="synergy_type",
        value_name="synergy_value",
    )

    red_df = df.melt(
        id_vars=[
            "E_group",
            "I_group",
            "cEE",
            "cII",
            "cEI",
            "model_type",
            "accuracy",
            "final_mi",
        ],
        value_vars=redundancy_cols,
        var_name="redundancy_type",
        value_name="redundancy_value",
    )

    # Plot 1: Synergy by group assignment and model type
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=syn_df, x="synergy_type", y="synergy_value", hue="model_type")
    plt.title("Synergy Distribution by Type and Model")
    plt.xlabel("Synergy Type")
    plt.ylabel("Synergy Value")
    output_path = os.path.join(RESULTS_FOLDER, f"{filename_prefix}_synergy_by_type.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved synergy plot to {output_path}")

    # Plot 2: Correlation between synergy and performance
    # Create separate plots for each model type to avoid label conflict
    for model_type in df["model_type"].unique():
        plt.figure(figsize=(14, 8))

        # Filter data for this model type
        syn_model_df = syn_df[syn_df["model_type"] == model_type]
        red_model_df = red_df[red_df["model_type"] == model_type]

        plt.subplot(1, 2, 1)
        sns.scatterplot(
            data=syn_model_df,
            x="synergy_value",
            y="accuracy",
            hue="synergy_type",
            alpha=0.7,
        )
        plt.title(f"Synergy vs Accuracy ({model_type} model)")
        plt.xlabel("Synergy Value")
        plt.ylabel("Accuracy")

        plt.subplot(1, 2, 2)
        sns.scatterplot(
            data=red_model_df,
            x="redundancy_value",
            y="accuracy",
            hue="redundancy_type",
            alpha=0.7,
        )
        plt.title(f"Redundancy vs Accuracy ({model_type} model)")
        plt.xlabel("Redundancy Value")
        plt.ylabel("Accuracy")

        plt.tight_layout()
        output_path = os.path.join(
            RESULTS_FOLDER,
            f"{filename_prefix}_performance_correlation_{model_type}.png",
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(
            f"Saved performance correlation plot for {model_type} to {output_path}"
        )

    # Combined plot with synergy types but without model separation
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=syn_df,
        x="synergy_value",
        y="accuracy",
        hue="synergy_type",
        style="model_type",
        alpha=0.7,
    )
    plt.title("Synergy vs Accuracy (All Models)")
    plt.xlabel("Synergy Value")
    plt.ylabel("Accuracy")
    output_path = os.path.join(
        RESULTS_FOLDER, f"{filename_prefix}_synergy_vs_accuracy.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved combined synergy plot to {output_path}")


def create_correlation_effect_plots(df, filename_prefix):
    """
    Create plots to visualize the effect of correlation parameters.

    Parameters:
    - df: DataFrame with the results
    - filename_prefix: Prefix for output filenames
    """
    # Convert correlation values to categorical for better visualization
    corr_df = df.copy()
    for col in ["cEE", "cII", "cEI"]:
        corr_df[f"{col}_cat"] = corr_df[col].apply(lambda x: f"{x:.1f}")

    # Plot 1: Effect of correlation parameters on accuracy by model type
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(data=corr_df, x="cEE_cat", y="accuracy", hue="model_type")
    plt.title("Effect of E-E Correlation on Accuracy")
    plt.xlabel("E-E Correlation")
    plt.ylabel("Accuracy")

    plt.subplot(1, 3, 2)
    sns.boxplot(data=corr_df, x="cII_cat", y="accuracy", hue="model_type")
    plt.title("Effect of I-I Correlation on Accuracy")
    plt.xlabel("I-I Correlation")
    plt.ylabel("Accuracy")

    plt.subplot(1, 3, 3)
    sns.boxplot(data=corr_df, x="cEI_cat", y="accuracy", hue="model_type")
    plt.title("Effect of E-I Correlation on Accuracy")
    plt.xlabel("E-I Correlation")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    output_path = os.path.join(
        RESULTS_FOLDER, f"{filename_prefix}_correlation_accuracy.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved correlation effect plot to {output_path}")

    # Plot 2: 3D scatter of correlation parameters
    for model_type in df["model_type"].unique():
        model_df = corr_df[corr_df["model_type"] == model_type]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            model_df["cEE"],
            model_df["cII"],
            model_df["cEI"],
            c=model_df["accuracy"],
            cmap="viridis",
            s=50,
            alpha=0.7,
        )

        ax.set_xlabel("E-E Correlation")
        ax.set_ylabel("I-I Correlation")
        ax.set_zlabel("E-I Correlation")
        ax.set_title(
            f"Correlation Parameter Space Colored by Accuracy ({model_type} model)"
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label("Accuracy")

        plt.tight_layout()
        output_path = os.path.join(
            RESULTS_FOLDER, f"{filename_prefix}_3d_correlation_{model_type}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved 3D correlation plot for {model_type} to {output_path}")


def create_group_information_plots(df, filename_prefix):
    """
    Create plots to visualize the effect of group information assignment.

    Parameters:
    - df: DataFrame with the results
    - filename_prefix: Prefix for output filenames
    """
    # Map group numbers to information content for better readability
    group_info_map = {0: "No Info", 1: "Medium Info", 2: "High Info"}
    plot_df = df.copy()

    plot_df["E_group_info"] = plot_df["E_group"].map(group_info_map)
    plot_df["I_group_info"] = plot_df["I_group"].map(group_info_map)

    # Plot 1: Heatmap of accuracy by E and I group information content
    for model_type in df["model_type"].unique():
        model_df = plot_df[plot_df["model_type"] == model_type]

        plt.figure(figsize=(10, 8))
        pivot = model_df.pivot_table(
            index="E_group_info",
            columns="I_group_info",
            values="accuracy",
            aggfunc="mean",
        )

        # Reorder index and columns to match information level
        pivot = pivot.reindex(group_info_map.values())
        pivot = pivot.reindex(columns=group_info_map.values())

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={"label": "Mean Accuracy"},
        )

        plt.title(f"Accuracy by E and I Group Information Content ({model_type} model)")
        plt.tight_layout()

        output_path = os.path.join(
            RESULTS_FOLDER, f"{filename_prefix}_group_info_heatmap_{model_type}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(
            f"Saved group information heatmap for {model_type} to {output_path}"
        )

    # Plot 2: Bar plot comparing synergy across group assignments
    for synergy_type in ["syn_ee", "syn_ii", "syn_ei"]:
        plt.figure(figsize=(12, 8))

        sns.barplot(data=plot_df, x="E_group_info", y=synergy_type, hue="I_group_info")

        plt.title(f"{synergy_type} by Group Information Content")
        plt.xlabel("E Group Information")
        plt.ylabel(f"Mean {synergy_type}")
        plt.xticks(rotation=0)
        plt.tight_layout()

        output_path = os.path.join(
            RESULTS_FOLDER, f"{filename_prefix}_{synergy_type}_by_group.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {synergy_type} by group plot to {output_path}")


def visualize_all_results(df):
    """
    Generate a comprehensive set of visualizations from the results dataframe.

    Parameters:
    - df: DataFrame with the simulation results
    """
    logger.info("Generating visualizations...")

    # Create output subfolder for plots
    plots_folder = os.path.join(RESULTS_FOLDER, "plots")
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    # 1. Basic performance visualization
    # 1.1 Scatterplot of MI vs. Accuracy by model type
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="final_mi", y="accuracy", hue="model_type", alpha=0.7)
    plt.title("Mutual Information vs. Accuracy")
    plt.xlabel("Mutual Information")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    output_path = os.path.join(plots_folder, "mi_vs_accuracy.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved MI vs. Accuracy plot to {output_path}")

    # 2. Group-based visualizations
    # 2.1 Heatmaps for accuracy by E_group and I_group
    create_heatmap(
        df,
        "E_group",
        "I_group",
        "accuracy",
        "Accuracy by E and I Group Assignment",
        "plots/accuracy_by_group_all.png",
    )

    for model in df["model_type"].unique():
        create_heatmap(
            df,
            "E_group",
            "I_group",
            "accuracy",
            "Accuracy by E and I Group Assignment",
            f"plots/accuracy_by_group_{model}.png",
            model_type=model,
        )

    # 2.2 Detailed group information plots
    create_group_information_plots(df, "plots/group_info")

    # 3. Correlation parameter visualizations
    create_correlation_effect_plots(df, "plots/correlation")

    # 4. Synergy and redundancy visualizations
    create_synergy_plots(df, "plots/synergy")

    # 5. Model comparison
    # 5.1 Box plot of accuracy by model type
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="model_type", y="accuracy")
    plt.title("Accuracy by Model Type")
    plt.xlabel("Model Type")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    output_path = os.path.join(plots_folder, "accuracy_by_model.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved accuracy by model plot to {output_path}")

    # 6. Pairplots for understanding metric relationships
    metric_vars = ["accuracy", "final_mi", "syn_ee", "syn_ii", "syn_ei"]
    create_pairplot(
        df,
        metric_vars,
        "model_type",
        "Relationships Between Performance Metrics",
        "plots/metric_relationships.png",
    )

    logger.info("All visualizations completed successfully")


def visualize_optimal_configurations(df, summary_df):
    """
    Create specialized plots focused on the best-performing configurations.

    Parameters:
    - df: DataFrame with all simulation results
    - summary_df: DataFrame with the best configurations identified
    """
    logger.info("Generating optimal configuration visualizations...")

    # Create output subfolder for optimal visualizations
    optimal_folder = os.path.join(RESULTS_FOLDER, "optimal_configs")
    if not os.path.exists(optimal_folder):
        os.makedirs(optimal_folder)

    # Extract top configurations for different metrics
    best_accuracy_config = summary_df.loc["Best Accuracy"]
    best_ee_synergy_config = summary_df.loc["Best E-E Synergy"]
    best_ei_synergy_config = summary_df.loc["Best E-I Synergy"]

    # 1. Compare optimal configurations side by side
    config_metrics = [
        "E_group",
        "I_group",
        "cEE",
        "cII",
        "cEI",
        "model_type",
        "accuracy",
        "final_mi",
        "syn_ee",
        "syn_ii",
        "syn_ei",
    ]

    opt_configs = pd.DataFrame(
        summary_df.loc[
            ["Best Accuracy", "Best E-E Synergy", "Best E-I Synergy"], config_metrics
        ]
    )

    # Create a radar plot for the optimal configurations
    metrics = ["accuracy", "final_mi", "syn_ee", "syn_ii", "syn_ei"]

    # Normalize metrics for radar plot
    for metric in metrics:
        max_val = df[metric].max()
        min_val = df[metric].min()
        opt_configs[f"{metric}_norm"] = (opt_configs[metric] - min_val) / (
            max_val - min_val
        )

    # Radar plot
    plt.figure(figsize=(10, 8))

    # Create angle array
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Set up the radar plot
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)  # Start from top
    ax.set_theta_direction(-1)  # Go clockwise

    # Add labels
    plt.xticks(angles[:-1], metrics)

    # Plot each configuration
    for idx, config_name in enumerate(
        ["Best Accuracy", "Best E-E Synergy", "Best E-I Synergy"]
    ):
        values = (
            opt_configs.loc[config_name, [f"{m}_norm" for m in metrics]]
            .values.flatten()
            .tolist()
        )
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=config_name)
        ax.fill(angles, values, alpha=0.1)

    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.title("Performance Metrics of Optimal Configurations")

    output_path = os.path.join(optimal_folder, "optimal_configs_radar.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved optimal configurations radar plot to {output_path}")

    # 2. Filter datasets for each optimal configuration and visualize in more detail
    # For each optimal configuration, find all entries with the same parameters
    for config_name, config in [
        ("best_accuracy", best_accuracy_config),
        ("best_ee_synergy", best_ee_synergy_config),
        ("best_ei_synergy", best_ei_synergy_config),
    ]:
        # Filter for configurations with same parameters
        mask = (
            (df["E_group"] == config["E_group"])
            & (df["I_group"] == config["I_group"])
            & (df["cEE"] == config["cEE"])
            & (df["cII"] == config["cII"])
            & (df["cEI"] == config["cEI"])
            & (df["model_type"] == config["model_type"])
        )

        similar_configs = df[mask].copy()

        if len(similar_configs) > 0:
            # Create a focused analysis for this optimal configuration
            plt.figure(figsize=(14, 10))

            # Plot 1: Synergy vs redundancy
            plt.subplot(2, 2, 1)
            plt.scatter(
                similar_configs["syn_ee"],
                similar_configs["red_ee"],
                label="E-E",
                alpha=0.7,
            )
            plt.scatter(
                similar_configs["syn_ii"],
                similar_configs["red_ii"],
                label="I-I",
                alpha=0.7,
            )
            plt.scatter(
                similar_configs["syn_ei"],
                similar_configs["red_ei"],
                label="E-I",
                alpha=0.7,
            )
            plt.axhline(0, color="gray", linestyle="--")
            plt.axvline(0, color="gray", linestyle="--")
            plt.xlabel("Synergy")
            plt.ylabel("Redundancy")
            plt.legend()
            plt.title(
                f'Synergy vs Redundancy for {config_name.replace("_", " ").title()}'
            )

            # Plot 2: Information distribution
            plt.subplot(2, 2, 2)
            info_level = ["No Info", "Medium Info", "High Info"]
            plt.bar(0, 1, label=f'E: {info_level[int(config["E_group"])]}')
            plt.bar(1, 1, label=f'I: {info_level[int(config["I_group"])]}')
            plt.xticks([0, 1], ["E Group", "I Group"])
            plt.ylabel("Information Level")
            plt.legend()
            plt.title(
                f'Information Distribution for {config_name.replace("_", " ").title()}'
            )

            # Plot 3: Correlation parameters
            plt.subplot(2, 2, 3)
            corr_labels = ["E-E", "I-I", "E-I"]
            corr_values = [config["cEE"], config["cII"], config["cEI"]]
            plt.bar(range(len(corr_labels)), corr_values)
            plt.axhline(0, color="gray", linestyle="--")
            plt.xticks(range(len(corr_labels)), corr_labels)
            plt.ylabel("Correlation Value")
            plt.title(
                f'Correlation Parameters for {config_name.replace("_", " ").title()}'
            )

            # Plot 4: Performance metrics
            plt.subplot(2, 2, 4)
            perf_labels = ["Accuracy", "MI", "Syn E-E", "Syn I-I", "Syn E-I"]
            perf_values = [
                config["accuracy"],
                config["final_mi"],
                config["syn_ee"],
                config["syn_ii"],
                config["syn_ei"],
            ]
            plt.bar(range(len(perf_labels)), perf_values)
            plt.xticks(range(len(perf_labels)), perf_labels, rotation=45)
            plt.ylabel("Value")
            plt.title(
                f'Performance Metrics for {config_name.replace("_", " ").title()}'
            )

            plt.tight_layout()
            output_path = os.path.join(optimal_folder, f"{config_name}_detailed.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved detailed analysis for {config_name} to {output_path}")

    logger.info("Optimal configuration visualizations completed.")


def run_ei_ratio_sweep(
    D=50, C=3, total_synapses=20, rho=0.3, seeds=[0], max_iter=20, apply_sigmoid=True
):
    """
    Sweep over different ratios of excitatory (N_E) to inhibitory (N_I) synapses
    while keeping the total number (N_E + N_I) fixed.

    Parameters:
    - D: Feature dimensions
    - C: Number of classes
    - total_synapses: Total number of synapses (N_E + N_I)
    - rho: Correlation parameter
    - seeds: Random seeds to use
    - max_iter: Maximum iterations for fitting
    - apply_sigmoid: Whether to apply sigmoid to inputs

    Returns:
    - DataFrame with results for all combinations
    """
    logger.info(
        f"Starting E/I ratio sweep with D={D}, C={C}, total_synapses={total_synapses}"
    )

    # Check if the total synapses can fit within a group
    dgrp = D // 3
    if total_synapses > 2 * dgrp:
        logger.error(
            f"Error: Total synapses ({total_synapses}) exceeds twice the group size ({dgrp})."
        )
        logger.error(
            "E and I dimensions might need to come from the same group, which would exceed the group size."
        )
        logger.error(f"Please ensure total_synapses <= {2 * dgrp}.")
        return pd.DataFrame()

    # Determine possible combinations of N_E and N_I
    possible_combos = []
    for n_e in range(1, total_synapses):
        n_i = total_synapses - n_e
        # Check if both N_E and N_I can fit in a group
        if n_e <= dgrp and n_i <= dgrp:
            possible_combos.append((n_e, n_i))

    if not possible_combos:
        logger.error(
            f"No valid N_E/N_I combinations found for total_synapses={total_synapses}"
        )
        return pd.DataFrame()

    logger.info(f"Testing {len(possible_combos)} different E/I ratios")

    # Run the simulation for each combination
    all_results = []
    for n_e, n_i in possible_combos:
        logger.info(f"Running with N_E={n_e}, N_I={n_i} (E/I ratio: {n_e/n_i:.2f})")
        df = run_full_enumeration(
            D=D,
            C=C,
            N_E=n_e,
            N_I=n_i,
            rho=rho,
            seeds=seeds,
            max_iter=max_iter,
            apply_sigmoid=apply_sigmoid,
        )

        if len(df) > 0:
            # Add E/I ratio information
            df["total_synapses"] = total_synapses
            df["ei_ratio"] = n_e / n_i
            df["e_fraction"] = n_e / total_synapses
            df["i_fraction"] = n_i / total_synapses
            all_results.append(df)

    # Combine results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        logger.info(
            f"E/I ratio sweep complete. Generated {len(combined_df)} total combinations."
        )
        return combined_df
    else:
        logger.error("No valid results generated in the E/I ratio sweep.")
        return pd.DataFrame()


def visualize_ei_ratio_results(df, output_dir=None):
    """
    Create visualizations to analyze the effect of E/I ratio on performance metrics.

    Parameters:
    - df: DataFrame with results from the E/I ratio sweep
    - output_dir: Directory to save plots (if None, uses RESULTS_FOLDER)
    """
    if output_dir is None:
        output_dir = os.path.join(RESULTS_FOLDER, "ei_ratio_analysis")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Generating E/I ratio analysis visualizations in {output_dir}")

    # Group by E/I ratio, model_type, and information groups
    agg_df = df.groupby(
        ["ei_ratio", "e_fraction", "i_fraction", "model_type", "E_group", "I_group"],
        as_index=False,
    ).agg(
        {
            "accuracy": "mean",
            "final_mi": "mean",
            "syn_ee": "mean",
            "syn_ii": "mean",
            "syn_ei": "mean",
            "red_ee": "mean",
            "red_ii": "mean",
            "red_ei": "mean",
        }
    )

    # 1. Accuracy vs E/I Ratio by model type
    plt.figure(figsize=(10, 6))
    for model in df["model_type"].unique():
        model_df = agg_df[agg_df["model_type"] == model]
        sns.lineplot(data=model_df, x="ei_ratio", y="accuracy", marker="o", label=model)

    plt.xscale("log")
    plt.xlabel("E/I Ratio (log scale)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs E/I Ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "accuracy_vs_ei_ratio.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved accuracy vs E/I ratio plot to {output_path}")

    # 2. Synergy vs E/I Ratio
    synergy_types = ["syn_ee", "syn_ii", "syn_ei"]
    plt.figure(figsize=(12, 6))

    for syn_type in synergy_types:
        sns.lineplot(
            data=agg_df,
            x="ei_ratio",
            y=syn_type,
            marker="o",
            label=syn_type.replace("syn_", ""),
        )

    plt.xscale("log")
    plt.xlabel("E/I Ratio (log scale)")
    plt.ylabel("Synergy")
    plt.title("Synergy vs E/I Ratio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "synergy_vs_ei_ratio.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved synergy vs E/I ratio plot to {output_path}")

    # 3. Heatmap of accuracy by E_group and I_group for best E/I ratio
    # Find best E/I ratio by accuracy
    best_ratio_idx = agg_df["accuracy"].idxmax()
    best_ratio = agg_df.loc[best_ratio_idx, "ei_ratio"]
    best_ratio_df = agg_df[np.isclose(agg_df["ei_ratio"], best_ratio)]

    plt.figure(figsize=(10, 8))
    pivot = best_ratio_df.pivot_table(
        index="E_group", columns="I_group", values="accuracy", aggfunc="mean"
    )

    # Map group numbers to information content
    group_info = {0: "No Info", 1: "Medium Info", 2: "High Info"}
    pivot.index = [group_info[i] for i in pivot.index]
    pivot.columns = [group_info[i] for i in pivot.columns]

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "Mean Accuracy"},
    )

    plt.title(f"Accuracy by E and I Information (Best E/I Ratio: {best_ratio:.2f})")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "best_ratio_accuracy_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved best E/I ratio accuracy heatmap to {output_path}")

    # 4. E/I Ratio vs Model Type interaction
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=agg_df,
        x="ei_ratio",
        y="accuracy",
        hue="model_type",
        style="model_type",
        markers=True,
        dashes=False,
    )

    plt.xscale("log")
    plt.xlabel("E/I Ratio (log scale)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs E/I Ratio by Model Type")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "ei_ratio_model_interaction.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved E/I ratio and model type interaction plot to {output_path}")

    # 5. Best E/I ratio for each information group combination
    group_best_ratio = (
        agg_df.groupby(["E_group", "I_group"], as_index=False)
        .apply(lambda x: x.loc[x["accuracy"].idxmax()])
        .reset_index(drop=True)
    )

    plt.figure(figsize=(12, 8))
    for e_group in sorted(df["E_group"].unique()):
        for i_group in sorted(df["I_group"].unique()):
            subset = group_best_ratio[
                (group_best_ratio["E_group"] == e_group)
                & (group_best_ratio["I_group"] == i_group)
            ]
            if len(subset) > 0:
                label = f"E: {group_info[e_group]}, I: {group_info[i_group]}"
                plt.scatter(
                    subset["ei_ratio"],
                    subset["accuracy"],
                    s=100,
                    label=label,
                    alpha=0.7,
                )

    plt.xscale("log")
    plt.xlabel("Best E/I Ratio (log scale)")
    plt.ylabel("Accuracy")
    plt.title("Best E/I Ratio for Each Information Group Combination")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "best_ratio_by_group.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved best E/I ratio by group plot to {output_path}")

    logger.info("E/I ratio analysis visualizations complete")
    return best_ratio


###############################################################################
# MAIN ENTRY POINT
###############################################################################

if __name__ == "__main__":
    # Example usage:
    # We'll do D=50 => each group has ~16-17 dims.
    # When picking dimensions from groups, ensure N_E and N_I are not larger than D/3

    logger.info("Starting dendritic network simulation")
    logger.info("======================================")

    # Configuration
    D = 30  # Feature dimensions
    C = 3  # Number of classes
    N_E = 10  # Number of excitatory dimensions (must be <= D/3)
    N_I = 2  # Number of inhibitory dimensions (must be <= D/3)
    rho = 0.2  # Correlation parameter
    seeds = [0]  # Random seeds
    max_iter = 50  # Maximum iterations for fitting
    apply_sigmoid = True  # Whether to apply sigmoid to inputs before processing

    logger.info(
        f"Configuration: D={D}, C={C}, N_E={N_E}, N_I={N_I}, rho={rho}, seeds={seeds}"
    )
    logger.info(f"Apply sigmoid normalization to inputs: {apply_sigmoid}")

    # Run the full enumeration
    start_time = time.time()
    df = run_full_enumeration(
        D=D,
        C=C,
        N_E=N_E,
        N_I=N_I,
        rho=rho,
        seeds=seeds,
        max_iter=max_iter,
        apply_sigmoid=apply_sigmoid,
    )

    total_time = time.time() - start_time
    logger.info(f"Full enumeration completed in {total_time:.2f} seconds")
    logger.info(f"Generated {len(df)} valid configurations")

    # Save the raw results to CSV
    out_csv = os.path.join(RESULTS_FOLDER, "all_9x27_combos.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved enumeration results to {out_csv}")

    # Handle the case where no valid configurations were found
    if len(df) == 0:
        logger.error(
            "No valid configurations were generated. Please adjust the parameters."
        )
        logger.error(f"Each group has approximately {D//3} dimensions.")
        logger.error(f"N_E ({N_E}) and N_I ({N_I}) must both be <= {D//3}.")
        sys.exit(1)

    # Create summary statistics
    logger.info("Generating summary statistics...")

    # Group by all configuration parameters and average across seeds
    pivot_df = df.groupby(
        ["E_group", "I_group", "cEE", "cII", "cEI", "model_type"], as_index=False
    ).agg(
        {
            "accuracy": "mean",
            "final_mi": "mean",
            "syn_ee": "mean",
            "syn_ii": "mean",
            "syn_ei": "mean",
            "red_ee": "mean",
            "red_ii": "mean",
            "red_ei": "mean",
        }
    )

    # Find best configurations
    best_accuracy_row = pivot_df.loc[pivot_df["accuracy"].idxmax()]
    best_synergy_ee_row = pivot_df.loc[pivot_df["syn_ee"].idxmax()]
    best_synergy_ei_row = pivot_df.loc[pivot_df["syn_ei"].idxmax()]

    logger.info("\nBest configuration by accuracy:")
    logger.info(
        f"  E_group: {best_accuracy_row['E_group']} (info level: {'none' if best_accuracy_row['E_group']==0 else 'medium' if best_accuracy_row['E_group']==1 else 'high'})"
    )
    logger.info(
        f"  I_group: {best_accuracy_row['I_group']} (info level: {'none' if best_accuracy_row['I_group']==0 else 'medium' if best_accuracy_row['I_group']==1 else 'high'})"
    )
    logger.info(
        f"  Correlations: E-E={best_accuracy_row['cEE']:.2f}, I-I={best_accuracy_row['cII']:.2f}, E-I={best_accuracy_row['cEI']:.2f}"
    )
    logger.info(f"  Model type: {best_accuracy_row['model_type']}")
    logger.info(f"  Accuracy: {best_accuracy_row['accuracy']:.4f}")

    # Save summary statistics
    summary_csv = os.path.join(RESULTS_FOLDER, "summary_statistics.csv")
    summary_df = pd.DataFrame(
        [best_accuracy_row, best_synergy_ee_row, best_synergy_ei_row],
        index=["Best Accuracy", "Best E-E Synergy", "Best E-I Synergy"],
    )
    summary_df.to_csv(summary_csv)
    logger.info(f"Saved summary statistics to {summary_csv}")

    try:
        # Generate comprehensive visualizations
        visualize_all_results(df)

        # Generate visualizations focused on optimal configurations
        visualize_optimal_configurations(df, summary_df)

        # Add required columns for E/I ratio visualization
        df["total_synapses"] = N_E + N_I
        df["ei_ratio"] = N_E / N_I
        df["e_fraction"] = N_E / (N_E + N_I)
        df["i_fraction"] = N_I / (N_E + N_I)

        # Generate E/I ratio analysis visualizations with the current simulation results
        logger.info("Generating E/I ratio analysis from current simulation...")
        best_ratio = visualize_ei_ratio_results(df)

        # Now run a dedicated E/I ratio sweep with varying proportions
        logger.info("Running E/I ratio sweep with fixed total synapses...")
        total_synapses = 12  # Total number of synapses (adjust as needed)
        ei_sweep_df = run_ei_ratio_sweep(
            D=D,
            C=C,
            total_synapses=total_synapses,
            rho=rho,
            seeds=seeds,
            max_iter=max_iter,
            apply_sigmoid=apply_sigmoid,
        )

        # Save E/I sweep results
        if len(ei_sweep_df) > 0:
            ei_sweep_csv = os.path.join(RESULTS_FOLDER, "ei_ratio_sweep.csv")
            ei_sweep_df.to_csv(ei_sweep_csv, index=False)
            logger.info(f"Saved E/I ratio sweep results to {ei_sweep_csv}")

            # Visualize E/I sweep results
            sweep_best_ratio = visualize_ei_ratio_results(
                ei_sweep_df,
                output_dir=os.path.join(RESULTS_FOLDER, "ei_ratio_sweep_analysis"),
            )
            logger.info(f"Best E/I ratio from sweep: {sweep_best_ratio:.2f}")
        else:
            logger.warning("E/I ratio sweep did not produce valid results.")

        logger.info("Simulation and analysis complete.")
    except Exception as e:
        logger.error(f"Error during visualization: {e!s}")
        import traceback

        logger.error(traceback.format_exc())

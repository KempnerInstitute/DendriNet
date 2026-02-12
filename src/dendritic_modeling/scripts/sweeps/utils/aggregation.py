"""Data aggregation utilities for sweep analysis."""

from typing import Optional

import numpy as np
import pandas as pd


def aggregate_over_seeds(
    data: pd.DataFrame,
    group_by_cols: list[str],
    metric_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate metrics over multiple seeds (compute mean and std).

    Args:
        data: DataFrame with results from multiple seeds
        group_by_cols: Columns to group by (e.g., ['ee_value', 'ie_value', 'use_shunting'])
        metric_cols: Columns to aggregate (if None, auto-detect)

    Returns:
        Aggregated DataFrame with _mean and _std columns
    """
    if data.empty:
        return data

    # Auto-detect metric columns if not provided
    if metric_cols is None:
        exclude_cols = {
            *group_by_cols,
            "seed",
            "task_id",
            "config_file",
            "job_dir",
            "experiment",
        }
        metric_cols = [col for col in data.columns if col not in exclude_cols]

    # Filter to only existing columns
    group_by_cols = [col for col in group_by_cols if col in data.columns]
    metric_cols = [col for col in metric_cols if col in data.columns]

    if not group_by_cols:
        return data

    # Group and aggregate
    results = []
    for group_vals, group_data in data.groupby(group_by_cols, dropna=False):
        # Create row with group values
        if isinstance(group_vals, tuple):
            row = dict(zip(group_by_cols, group_vals))
        else:
            row = {group_by_cols[0]: group_vals}

        # Aggregate each metric
        for metric in metric_cols:
            if metric not in group_data.columns:
                continue

            values = group_data[metric].dropna()
            if len(values) > 0:
                try:
                    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
                    if len(numeric_values) > 0:
                        row[f"{metric}_mean"] = np.mean(numeric_values)
                        row[f"{metric}_std"] = np.std(numeric_values)
                except (ValueError, TypeError):
                    # Categorical data - keep first value
                    row[metric] = values.iloc[0]

        results.append(row)

    return pd.DataFrame(results)


def group_by_params(
    data: pd.DataFrame,
    param_cols: list[str],
) -> pd.DataFrame:
    """
    Group data by parameter combinations.

    Args:
        data: DataFrame with sweep results
        param_cols: Parameter columns to group by

    Returns:
        DataFrame with one row per unique parameter combination
    """
    if data.empty or not param_cols:
        return data

    # Filter to only existing columns
    param_cols = [col for col in param_cols if col in data.columns]

    if not param_cols:
        return data

    # Get unique parameter combinations
    return (
        data[param_cols]
        .drop_duplicates()
        .sort_values(by=param_cols)
        .reset_index(drop=True)
    )


def compute_ei_ratio(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute E/I ratio from ee_value and ie_value columns.

    Args:
        data: DataFrame with ee_value and ie_value columns

    Returns:
        DataFrame with ei_ratio column added
    """
    if "ee_value" not in data.columns or "ie_value" not in data.columns:
        return data

    result = data.copy()

    # Handle cases where ie_value might be zero
    valid_mask = (
        (result["ie_value"] > 0)
        & (result["ie_value"].notna())
        & (result["ee_value"].notna())
    )
    result.loc[valid_mask, "ei_ratio"] = (
        result.loc[valid_mask, "ee_value"] / result.loc[valid_mask, "ie_value"]
    )

    return result


def extract_network_category(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract network category from use_shunting column.

    Args:
        data: DataFrame with use_shunting column

    Returns:
        DataFrame with network_category column added
    """
    if "use_shunting" not in data.columns:
        return data

    result = data.copy()
    result["network_category"] = "unknown"
    
    result.loc[
        (result["network_type"] == "einet") & (result["use_shunting"] == True), 
        "network_category"
    ] = "einet_sh"
    
    result.loc[
        (result["network_type"] == "einet") & (result["use_shunting"] == False), 
        "network_category"
    ] = "einet_ns"

    result.loc[
        result["network_type"] == "einet_sh", 
        "network_category"
    ] = "einet_sh"
    
    result.loc[
        result["network_type"] == "einet_ns", 
        "network_category"
    ] = "einet_ns"

    result.loc[
        result["network_type"] == "einet_sh_flat", 
        "network_category"
    ] = "einet_sh_flat"
    
    result.loc[
        result["network_type"] == "einet_ns_flat", 
        "network_category"
    ] = "einet_ns_flat"

    result.loc[
        result["network_type"] == "matchedtotalparammlp", 
        "network_category"
    ] = "total_param_mlp"

    result.loc[
        result["network_type"] == "matchedactiveparammlp", 
        "network_category"
    ] = "active_param_mlp"

    result.loc[
        result["network_type"] == "ss_mlp", 
        "network_category"
    ] = "ss_mlp"

    result.loc[
        result["network_type"] == "ss_mlp_flat", 
        "network_category"
    ] = "ss_mlp_flat"

    return result

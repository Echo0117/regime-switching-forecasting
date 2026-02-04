#!/usr/bin/env python3
"""
Run AgACI on multiple models separately and aggregate results.

This script:
1. Runs test_agaci.py for each model individually with --save-coverage-csv
2. Aggregates coverage/length data from all models
3. Plots the averaged multi-model results

Usage:
    python experiments/run_multimodel_agaci.py --problem Toy --n-models 10
    python experiments/run_multimodel_agaci.py --problem Toy --n-models 10 --data-dir Deep_Switching_State_Space_Model/data/Toy_exp2_V1_0.5_V2_1.0
    python experiments/run_multimodel_agaci.py --problem Toy --n-models 10 --use-oracle-switches
"""

import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path
import glob

def run_model(model_idx, args):
    """Run test_agaci.py for a specific model and return success status."""
    print(f"\n{'='*60}")
    print(f"Running AgACI for Model {model_idx}")
    print(f"{'='*60}")

    cmd = [
        "python", "experiments/test_agaci.py",
        "--problem", args.problem,
        "--model-idx", str(model_idx),
        "--alpha", str(args.alpha),
        "--save-coverage-csv",
        "--d-dim", str(args.d_dim),
    ]

    if args.data_dir:
        cmd.extend(["--data-dir", args.data_dir])

    if args.use_oracle_switches:
        cmd.append("--use-oracle-switches")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Model {model_idx} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Model {model_idx} failed:")
        print(e.stderr)
        return False

def filter_models_by_switches(args, max_switches=5):
    """
    Check each model's regime switches and return list of valid model indices.

    Returns:
        list: Model indices with <= max_switches regime switches
    """
    import torch
    import sys as _sys

    # Add paths for imports
    HERE = os.path.dirname(__file__)
    PROJ = os.path.abspath(os.path.join(HERE, ".."))
    for p in [HERE, PROJ, 'Deep_Switching_State_Space_Model/src']:
        if p not in _sys.path:
            _sys.path.insert(0, p)

    from experiments.utils.ds3m_utils import load_ds3m_data, forecast
    from DSSSMCode import DSSSM

    print(f"\n{'='*60}")
    print(f"Filtering models by regime switches (max: {max_switches})")
    print(f"{'='*60}\n")

    # Load dataset info
    ds = load_ds3m_data(args)

    valid_models = []
    skipped_models = []

    model_idx = 0
    max_models_to_try = args.n_models * 2  # Try up to 2x in case some are skipped

    while len(valid_models) < args.n_models and model_idx < max_models_to_try:
        model_path = os.path.join(ds["directoryBest"], f"best_model{model_idx}.tar")

        if not os.path.exists(model_path):
            print(f"Model {model_idx}: File not found, skipping")
            model_idx += 1
            continue

        # Load model
        model = DSSSM(
            ds["x_dim"], ds["y_dim"], ds["h_dim"], ds["z_dim"],
            ds["d_dim"], ds["n_layers"], ds["device"], ds["bidirection"]
        ).to(ds["device"])

        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])

        # Get forecasts to extract regime switches
        res, _, _, _, d_argmax_test, _, _ = forecast(
            model,
            ds["testX"], ds["testY"],
            ds["moments"], ds["d_dim"],
            ds["means"], ds["trend"],
            ds["test_len"], ds["freq"],
            ds["RawDataOriginal"],
            remove_mean=ds["remove_mean"],
            remove_residual=ds["remove_residual"],
        )

        # Count regime switches
        n_switches = np.sum(np.diff(d_argmax_test) != 0)

        if n_switches <= max_switches:
            valid_models.append(model_idx)
            print(f"Model {model_idx}: ✓ Valid ({n_switches} switches)")
        else:
            skipped_models.append(model_idx)
            print(f"Model {model_idx}: ✗ Skipped ({n_switches} switches > {max_switches})")

        model_idx += 1

    print(f"\n{'='*60}")
    print(f"Model filtering complete:")
    print(f"  Valid models: {len(valid_models)} - {valid_models}")
    print(f"  Skipped models: {len(skipped_models)} - {skipped_models}")
    print(f"{'='*60}\n")

    if len(valid_models) < args.n_models:
        print(f"⚠️  Warning: Only found {len(valid_models)} valid models (requested {args.n_models})")
        print(f"Continuing with {len(valid_models)} models...")

    return valid_models

def aggregate_results(csv_files):
    """
    Aggregate coverage/length data from multiple model CSVs.

    Returns:
        pd.DataFrame: Aggregated data with averaged coverage/length across models
    """
    print(f"\n{'='*60}")
    print(f"Aggregating results from {len(csv_files)} models")
    print(f"{'='*60}\n")

    # Read all CSVs
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                print(f"  Warning: {csv_file} is empty (no regime switches), skipping")
                continue
            dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"  Warning: {csv_file} is empty (no regime switches), skipping")
            continue

    if len(dfs) == 0:
        raise ValueError("All CSV files are empty! No models had regime switches in the test set.")

    # Get method names from first dataframe
    first_df = dfs[0]
    method_names = []
    for col in first_df.columns:
        if col.endswith('_coverage'):
            method_name = col.replace('_coverage', '')
            method_names.append(method_name)

    print(f"Found methods: {method_names}")

    # Concatenate all dataframes and group by (switch_idx, t_relative)
    # This handles cases where different models have different numbers of switches
    all_data = pd.concat(dfs, ignore_index=True)

    print(f"  Total rows before aggregation: {len(all_data)}")
    print(f"  Unique (switch_idx, t_relative) pairs: {all_data.groupby(['t_relative']).size().shape[0]}")

    # Group by t_relative (time relative to switch) and compute mean
    # We average across all switches and all models
    aggregated_data = all_data.groupby('t_relative').mean().reset_index()

    print(f"✓ Aggregated {len(aggregated_data)} rows (unique t_relative values)")

    # Keep only coverage and length columns, plus t_relative
    cols_to_keep = ['t_relative']
    for method_name in method_names:
        cols_to_keep.append(f'{method_name}_coverage')
        cols_to_keep.append(f'{method_name}_length')

    aggregated_data = aggregated_data[cols_to_keep]

    return aggregated_data, method_names

def plot_aggregated_coverage(aggregated_data, method_names, save_path):
    """Plot averaged coverage at regime switches."""
    print(f"\nPlotting aggregated coverage...")

    # Group by relative time and average across all switches
    grouped = aggregated_data.groupby('t_relative').mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    for method_name in method_names:
        coverage_col = f'{method_name}_coverage'
        if coverage_col in grouped.columns:
            ax.plot(grouped.index, grouped[coverage_col],
                   marker='o', label=method_name, linewidth=2)

    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Switch point')
    ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='Target (90%)')

    ax.set_xlabel('Time relative to switch', fontsize=12)
    ax.set_ylabel('Coverage (fraction of models)', fontsize=12)
    ax.set_title('Multi-Model Coverage at Regime Switches', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {save_path}")
    plt.close()

def plot_aggregated_length(aggregated_data, method_names, save_path):
    """Plot averaged interval length at regime switches."""
    print(f"\nPlotting aggregated interval length...")

    # Group by relative time and average across all switches
    grouped = aggregated_data.groupby('t_relative').mean()

    fig, ax = plt.subplots(figsize=(12, 6))

    for method_name in method_names:
        length_col = f'{method_name}_length'
        if length_col in grouped.columns:
            ax.plot(grouped.index, grouped[length_col],
                   marker='o', label=method_name, linewidth=2)

    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Switch point')

    ax.set_xlabel('Time relative to switch', fontsize=12)
    ax.set_ylabel('Interval length', fontsize=12)
    ax.set_title('Multi-Model Interval Length at Regime Switches', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {save_path}")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", type=str, default="Toy")
    ap.add_argument("--n-models", type=int, default=10)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--d-dim", type=int, default=2)
    ap.add_argument("--use-oracle-switches", action="store_true", default=False)
    ap.add_argument("--max-switches", type=int, default=2,
                    help="Maximum allowed switches per model")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (not used for filtering, just for load_ds3m_data)")
    args = ap.parse_args()

    print("="*60)
    print(f"Running multi-model AgACI for {args.problem}")
    print(f"Number of models: {args.n_models}")
    if args.data_dir:
        print(f"Data directory: {args.data_dir}")
    print(f"Max switches per model: {args.max_switches}")
    print("="*60)

    # Step 1: Filter models by regime switches
    valid_model_indices = filter_models_by_switches(args, max_switches=args.max_switches)

    if len(valid_model_indices) == 0:
        print("ERROR: No valid models found!")
        return

    # Step 2: Run test_agaci.py for each valid model
    successful_models = []
    for model_idx in valid_model_indices:
        success = run_model(model_idx, args)
        if success:
            successful_models.append(model_idx)

    print(f"\n{'='*60}")
    print(f"Completed {len(successful_models)}/{len(valid_model_indices)} models successfully")
    print(f"{'='*60}\n")

    if len(successful_models) == 0:
        print("ERROR: No models completed successfully!")
        return

    # Step 3: Find and aggregate CSV files
    # Determine dataset identifier
    if args.data_dir:
        dataset_id = os.path.basename(args.data_dir)
    else:
        dataset_id = args.problem

    # Find CSV files for successful models
    csv_pattern = f"figures/agaci_test/{dataset_id}_*/coverage_data_model*.csv"
    all_csv_files = glob.glob(csv_pattern)

    # Filter to only include successful models
    csv_files = []
    for model_idx in successful_models:
        matching = [f for f in all_csv_files if f"_model{model_idx}.csv" in f]
        if matching:
            csv_files.append(matching[0])

    print(f"\nFound {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")

    if len(csv_files) == 0:
        print("ERROR: No CSV files found!")
        return

    # Step 4: Aggregate results
    aggregated_data, method_names = aggregate_results(csv_files)

    # Step 5: Save aggregated CSV
    output_dir = f"figures/agaci_test/{dataset_id}_multimodel_n{len(csv_files)}"
    os.makedirs(output_dir, exist_ok=True)

    aggregated_csv = f"{output_dir}/coverage_data_aggregated.csv"
    aggregated_data.to_csv(aggregated_csv, index=False)
    print(f"\n✓ Saved aggregated data to {aggregated_csv}")

    # Step 6: Generate plots
    plot_aggregated_coverage(
        aggregated_data, method_names,
        f"{output_dir}/coverage_at_switches_aggregated.png"
    )

    plot_aggregated_length(
        aggregated_data, method_names,
        f"{output_dir}/length_at_switches_aggregated.png"
    )

    print(f"\n{'='*60}")
    print("✓ Multi-model AgACI analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

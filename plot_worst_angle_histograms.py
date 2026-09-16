"""
plot_worst_angle_histograms.py

Scans a directory for the CSV files produced by json_to_csv.py (i.e.
results_10km.csv, results_100km.csv, etc.) and plots a histogram of the
"worst_angle_deg" column for each one - a distribution of the
most-obstructed horizon angle across all sampled grid points, per
resolution/run.

Also produces one combined overlay plot (all runs on the same axes,
semi-transparent) so you can visually compare how the distribution
shifts as stride/resolution changes.

Requires matplotlib and pandas (visualization/analysis script, meant to
run on your normal dev machine, not the closed environment).

Usage:
    # Scan current directory for results_*.csv, plot each + a combined overlay
    python plot_worst_angle_histograms.py

    # Scan a specific directory, custom filename pattern
    python plot_worst_angle_histograms.py --directory "C:\\Users\\Greg\\Lunar Paper" --pattern "results_*.csv"

    # Skip the combined overlay, just individual histograms
    python plot_worst_angle_histograms.py --no_combined

    # Customize bin count
    python plot_worst_angle_histograms.py --bins 50
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_single_histogram(csv_path, column, bins, output_dir):
    df = pd.read_csv(csv_path)

    if column not in df.columns:
        print(f"  Skipping {csv_path} - no '{column}' column found "
              f"(columns present: {list(df.columns)})")
        return None

    values = df[column].dropna()
    if len(values) == 0:
        print(f"  Skipping {csv_path} - '{column}' column is empty/all-NaN")
        return None

    basename = os.path.splitext(os.path.basename(csv_path))[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(values, bins=bins, color="steelblue", edgecolor="black")
    ax.set_xlabel("Worst-Case Minimum Elevation Angle Per Target")
    ax.set_ylabel("Count")
    ax.set_title(f"Worst-Case Minimum Elevation Angle Per Target: {basename}  (n={len(values)})")

    out_path = os.path.join(output_dir, f"{basename}_worst_angle_hist.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"  {csv_path}: n={len(values)}, "
          f"mean={values.mean():.2f}, min={values.min():.2f}, max={values.max():.2f} "
          f"-> {out_path}")

    return basename, values


def plot_combined_overlay(all_series, bins, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))

    for basename, values in all_series:
        ax.hist(values, bins=bins, alpha=0.5, label=basename, edgecolor="none")

    ax.set_xlabel("Worst-Case Minimum Elevation Angle Per Target")
    ax.set_ylabel("Count")
    ax.set_title("Worst-Case Minimum Elevation Angle Per Target")
    ax.legend()

    out_path = os.path.join(output_dir, "combined_worst_angle_hist.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\nCombined overlay -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms of worst_angle_deg from results CSV files."
    )
    parser.add_argument("--directory", default=".",
                         help="Directory to scan for CSV files (default: current directory)")
    parser.add_argument("--pattern", default="results_*.csv",
                         help="Filename glob pattern (default: results_*.csv)")
    parser.add_argument("--column", default="worst_angle_deg",
                         help="Column to histogram (default: worst_angle_deg)")
    parser.add_argument("--bins", type=int, default=30,
                         help="Number of histogram bins (default: 30)")
    parser.add_argument("--output_dir", default=None,
                         help="Where to save PNGs (default: same as --directory)")
    parser.add_argument("--no_combined", action="store_true",
                         help="Skip the combined overlay plot")

    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir else args.directory

    search_path = os.path.join(args.directory, args.pattern)
    csv_files = sorted(glob.glob(search_path))

    if not csv_files:
        print(f"No files matched {search_path}")
        return

    print(f"Found {len(csv_files)} file(s) matching {search_path}:")
    for f in csv_files:
        print(f"  {f}")
    print()

    all_series = []
    for csv_path in csv_files:
        result = plot_single_histogram(csv_path, args.column, args.bins, output_dir)
        if result is not None:
            all_series.append(result)

    if not args.no_combined and len(all_series) > 1:
        plot_combined_overlay(all_series, args.bins, output_dir)


if __name__ == "__main__":
    main()

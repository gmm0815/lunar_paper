"""
json_to_csv.py

Converts the JSON output of full_grid_horizon_masks.py (results_10km.json,
results_100km.json, etc.) into CSV files, via a pandas DataFrame.

Each row in the CSV corresponds to one grid point, with columns matching
the per-point fields written by full_grid_horizon_masks.py: row, col,
lat, lon, h_target, worst_azimuth, worst_angle_deg, best_azimuth,
best_angle_deg (and full_profile_deg, if that run was made with
--save_full_profile - see the note in convert_results_json_to_csv below).

Usage as a library:
    from json_to_csv import convert_results_json_to_csv
    convert_results_json_to_csv("results_100km.json", "results_100km.csv")

Usage from the command line (converts one or more files at once):
    python json_to_csv.py results_10km.json results_100km.json results_1000km.json
    python json_to_csv.py results_25km.json --output custom_name.csv
"""

import os
import json
import argparse
import pandas as pd


def convert_results_json_to_csv(json_path, csv_path=None):
    """
    Load a full_grid_horizon_masks.py results JSON file and write its
    per-point results out as a CSV via pandas.

    Parameters
    ----------
    json_path : str
        Path to the input results JSON file.
    csv_path : str, optional
        Path for the output CSV. Defaults to the same basename as
        json_path, with a .csv extension.

    Returns
    -------
    pandas.DataFrame
        The DataFrame that was written to csv_path, in case you want to
        keep working with it in-memory (e.g. in a notebook) without
        re-reading the file back in.
    """
    if csv_path is None:
        csv_path = os.path.splitext(json_path)[0] + ".csv"

    with open(json_path, "r") as f:
        data = json.load(f)

    results = data["results"]
    df = pd.DataFrame(results)

    # Note: if this run was made with --save_full_profile, each row's
    # "full_profile_deg" field is a 360-length list. pandas will store
    # that as a single object column (each cell holding a Python list),
    # which round-trips through CSV as a string rather than 360 separate
    # numeric columns. That's usually fine for archival, but if you want
    # it usably numeric again after reloading, you'd need something like:
    #   import ast
    #   df["full_profile_deg"] = df["full_profile_deg"].apply(ast.literal_eval)
    # after re-reading the CSV.

    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows -> {csv_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert full_grid_horizon_masks.py JSON results into CSV files."
    )
    parser.add_argument("json_files", nargs="+",
                         help="One or more results JSON files to convert")
    parser.add_argument("--output", default=None,
                         help="Output CSV path. Only valid when converting a "
                              "single input file - for multiple inputs, each "
                              "gets its own auto-named .csv alongside it.")

    args = parser.parse_args()

    if args.output and len(args.json_files) > 1:
        parser.error("--output can only be used with a single input file")

    for json_path in args.json_files:
        csv_path = args.output if args.output else None
        convert_results_json_to_csv(json_path, csv_path)

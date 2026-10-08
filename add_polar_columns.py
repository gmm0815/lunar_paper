"""
add_polar_columns.py

Reads a results file (CSV or the JSON written by the horizon-mask scripts), converts
each row's latitude/longitude into SOUTH-POLAR STEREOGRAPHIC plane coordinates (the
frame used by the LOLA / LPI South Pole Atlas maps), and writes a copy of the data
with the new columns appended. Every original column is kept unchanged; the input
file is never modified.

New columns:
    x_km       east-west position in km from the pole (0 deg longitude is "up",
               90 E is to the right), true scale at the pole
    y_km       north-south position in km from the pole (positive toward 0 deg lon)
    rho_km     distance from the pole in the plane (km)
    azimuth_deg  longitude measured from the "up" direction, 0..360

With these, a plain scatter of x_km vs y_km IS the polar map - no further projection:
    plt.scatter(df.x_km, df.y_km, c=df.my_column)

Formula (sphere R = 1737.4 km):
    rho = 2 R tan((90 deg + lat) / 2)
    x = rho sin(lon - center_lon)        y = rho cos(lon - center_lon)

Rows north of --max_lat (default 0) get blank x/y (the south-polar projection does
not make sense there; it diverges toward the north pole), and so do rows with a
missing lat/lon.

Usage:
    python add_polar_columns.py my_results.csv
        -> my_results_polar.csv
    python add_polar_columns.py my_results.json --output with_xy.csv
    python add_polar_columns.py my_results.csv --lat_col latitude --lon_col longitude
    python add_polar_columns.py my_results.csv --center_lon 90 --mirror

    --center_lon   longitude that should point UP in the plane (default 0)
    --mirror       flip east/west (only if an image you overlay is mirrored)

If the file already has x_km / y_km (e.g. from polar_horizon_masks.py) they are
NOT replaced: they come from the DEM's own frame. Pass --overwrite to recompute.
Output files over 24 MB can be split with split_csv.py.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

R_MOON_KM = 1737.4
NEW_COLUMNS = ["x_km", "y_km", "rho_km", "azimuth_deg"]


def project(lat_deg, lon_deg, center_lon=0.0, mirror=False, max_lat=0.0):
    """Vectorised south-polar stereographic projection; NaN where undefined."""
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon) & (lat <= max_lat) & (lat >= -90.0)
    rho = np.full(lat.shape, np.nan)
    rho[ok] = 2.0 * R_MOON_KM * np.tan(np.radians(90.0 + lat[ok]) / 2.0)
    lam = np.radians(lon - center_lon)
    x = rho * np.sin(lam)
    y = rho * np.cos(lam)
    if mirror:
        x = -x
    azimuth = np.degrees(np.arctan2(x, y)) % 360.0
    azimuth[~ok] = np.nan
    return x, y, rho, azimuth


def read_table(path):
    if path.lower().endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        rows = data["results"] if isinstance(data, dict) and "results" in data else data
        # keep scalar fields only (drops e.g. full_profile_deg lists)
        keep = [k for k, v in rows[0].items() if not isinstance(v, (list, dict))]
        return pd.DataFrame([{k: r.get(k) for k in keep} for r in rows])
    return pd.read_csv(path)


def main():
    p = argparse.ArgumentParser(description="Append south-polar-plane x/y columns to a results file.")
    p.add_argument("input", help="CSV or results JSON containing latitude and longitude columns")
    p.add_argument("--output", default=None, help="Output CSV (default: <input>_polar.csv)")
    p.add_argument("--lat_col", default=None, help="Latitude column name (default: lat / latitude)")
    p.add_argument("--lon_col", default=None, help="Longitude column name (default: lon / longitude)")
    p.add_argument("--center_lon", type=float, default=0.0)
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--max_lat", type=float, default=0.0,
                   help="Rows north of this latitude get blank x/y (default 0)")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute x_km/y_km even if the file already has them")
    args = p.parse_args()

    df = read_table(args.input)

    def pick(given, options, what):
        if given:
            if given not in df.columns:
                raise SystemExit(f"Column '{given}' not found. Columns: {list(df.columns)}")
            return given
        for c in options:
            if c in df.columns:
                return c
        raise SystemExit(f"No {what} column found (tried {options}). Columns: {list(df.columns)}. "
                         f"Pass its name with --{what[:3]}_col.")

    lat_col = pick(args.lat_col, ["lat", "latitude", "Lat", "Latitude", "LAT"], "latitude")
    lon_col = pick(args.lon_col, ["lon", "longitude", "Lon", "Longitude", "LON"], "longitude")

    present = [c for c in NEW_COLUMNS if c in df.columns]
    if present and not args.overwrite:
        raise SystemExit(f"The file already has {present}. They were probably computed in the DEM's "
                         f"own polar frame - keep them, or pass --overwrite to recompute.")
    if present:
        df = df.drop(columns=present)

    x, y, rho, az = project(df[lat_col], df[lon_col], args.center_lon, args.mirror, args.max_lat)
    df["x_km"], df["y_km"], df["rho_km"], df["azimuth_deg"] = x, y, rho, az

    out = args.output or f"{os.path.splitext(args.input)[0]}_polar.csv"
    if os.path.abspath(out) == os.path.abspath(args.input):
        raise SystemExit("Refusing to overwrite the input file; choose a different --output.")
    df.to_csv(out, index=False)

    n_ok = int(np.isfinite(x).sum())
    print(f"Read {len(df)} rows; projected {n_ok} ({len(df) - n_ok} left blank: "
          f"missing lat/lon or north of {args.max_lat:g} deg).")
    if n_ok:
        print(f"  x_km {np.nanmin(x):.2f} .. {np.nanmax(x):.2f}   y_km {np.nanmin(y):.2f} .. {np.nanmax(y):.2f}   "
              f"farthest point {np.nanmax(rho):.1f} km from the pole")
    print(f"  orientation: {args.center_lon:g} deg longitude up" + (", mirrored" if args.mirror else ""))
    print(f"Saved {out}  ({os.path.getsize(out) / 1e6:.2f} MB)")
    if os.path.getsize(out) > 24e6:
        print("  Note: over 24 MB - split it with  python split_csv.py " + out)


if __name__ == "__main__":
    main()

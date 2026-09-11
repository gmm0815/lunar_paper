"""
full_grid_horizon_masks.py

Computes horizon-mask summaries (worst obstruction angle + which azimuth
it occurs at, best/clearest azimuth) for coordinates across the FULL LOLA
grid, rather than a single point.

IMPORTANT SCALE WARNING - read before running:
Your grid is 5120 x 10240 = ~52 million points. A full 360-azimuth x
500km horizon scan at every single point is not computationally
realistic (likely days-to-weeks depending on hardware), and storing a
full 360-value profile per point would need on the order of ~75GB.

This script addresses that with:
  - --stride: only process every Nth row/column (default 20 -> ~131k
    points instead of 52 million). Increase this further if the printed
    time estimate is still too large for you.
  - A timing pass BEFORE the real run: times a small sample of points,
    extrapolates to the full requested run, and asks for confirmation
    before proceeding (skip with --yes).
  - Per-point output is a SUMMARY (worst azimuth/angle, best azimuth/angle)
    rather than the full 360-value profile, to keep output size sane.
    Use --save_full_profile if you specifically need the full profile for
    a (presumably much more strided/limited) subset of points.
  - Incremental saving every --checkpoint_every points, so an interrupted
    run doesn't lose everything.
  - Skips points that are nodata (NaN) in the DEM - no elevation, no
    computation needed.

Loads the LunarDEM ONCE and reuses it across all points, rather than
re-opening chunk files per point (which the single-point get_horizon_mask
does, since it's designed for one-off calls).

Usage:
    python full_grid_horizon_masks.py ldem_1km results_grid
    python full_grid_horizon_masks.py ldem_1km results_grid --stride 10 --max_radius_km 500
    python full_grid_horizon_masks.py ldem_1km results_grid --stride 5 --yes
"""

import sys
import time
import json
import argparse
import numpy as np
from lola_dem_reader import LunarDEM
from coordinates_to_min_el import translate_lunar_coordinate, R_MOON_M


def colrow_to_lonlat(row, col, transform):
    """
    Inverse of LunarDEM._lonlat_to_colrow: given a pixel's (row, col),
    return the (lon, lat) in degrees of that pixel's CENTER, using the
    same Simple Cylindrical / Equirectangular projection.
    """
    a, b, c, d, e, f = transform
    # pixel center (+0.5 in each axis)
    x = a * (col + 0.5) + b * (row + 0.5) + c
    y = d * (col + 0.5) + e * (row + 0.5) + f
    lon = np.degrees(x / R_MOON_M)
    lat = np.degrees(y / R_MOON_M)
    return lat, lon


def horizon_mask_for_point(dem, target_lat, target_lon, h_target,
                            max_radius_km=500, num_points=360):
    """
    Same math as get_horizon_mask() in coordinates_to_min_el.py, but takes
    an already-loaded LunarDEM (and already-known h_target) instead of a
    basename, to avoid re-opening chunk files on every call.
    """
    horizon_angles = np.full(num_points, np.nan)
    azimuths = np.arange(0, num_points)

    for i, azimuth in enumerate(azimuths):
        distances_km = np.arange(1, max_radius_km + 1)

        lats2, lons2 = translate_lunar_coordinate(
            target_lat, target_lon, distances_km, bearing_deg=azimuth
        )

        coords = list(zip(lons2, lats2))
        h_terrain = dem.sample_many(coords)

        delta_h = h_terrain - h_target
        d = distances_km * 1000.0

        angle_rad = np.arctan2(delta_h - (d ** 2 / (2 * R_MOON_M)), d)
        angle_deg = np.degrees(angle_rad)

        if np.all(np.isnan(angle_deg)):
            horizon_angles[i] = np.nan
        else:
            horizon_angles[i] = np.nanmax(angle_deg)

    return horizon_angles  # length num_points, index == azimuth


def estimate_runtime(dem, sample_points, max_radius_km, num_points):
    """Time a handful of representative points, return seconds/point."""
    if not sample_points:
        return None

    timings = []
    for (row, col, lat, lon, h_target) in sample_points:
        t0 = time.time()
        horizon_mask_for_point(dem, lat, lon, h_target,
                                max_radius_km=max_radius_km, num_points=num_points)
        timings.append(time.time() - t0)

    return float(np.mean(timings))


def run_full_grid(dem_basename, output_basename, stride=20, max_radius_km=500,
                   num_points=360, checkpoint_every=500, save_full_profile=False,
                   skip_confirmation=False, n_timing_samples=5):
    dem = LunarDEM(dem_basename)
    n_rows, n_cols = dem.shape
    transform = dem.transform

    # --- Build the list of (row, col) points to process, honoring stride ---
    rows_to_check = range(0, n_rows, stride)
    cols_to_check = range(0, n_cols, stride)

    candidate_points = []
    for row in rows_to_check:
        for col in cols_to_check:
            lat, lon = colrow_to_lonlat(row, col, transform)
            h_target = dem.sample(lon, lat)
            if h_target is None:
                continue  # nodata - skip, nothing to compute
            candidate_points.append((row, col, lat, lon, h_target))

    total_points = len(candidate_points)
    print(f"Grid: {n_rows} x {n_cols}, stride={stride} -> "
          f"{len(rows_to_check)} x {len(cols_to_check)} candidate positions")
    print(f"Valid (non-nodata) points to process: {total_points}")

    if total_points == 0:
        print("Nothing to do - no valid points found. Check your stride/DEM.")
        return

    # --- Time a small sample, extrapolate, and confirm before the real run ---
    timing_sample = candidate_points[:min(n_timing_samples, total_points)]
    sec_per_point = estimate_runtime(dem, timing_sample, max_radius_km, num_points)
    est_total_seconds = sec_per_point * total_points

    print(f"\nTiming sample: {sec_per_point:.3f} sec/point "
          f"(measured on {len(timing_sample)} points)")
    print(f"Estimated total runtime: {est_total_seconds:.0f} sec "
          f"(~{est_total_seconds / 60:.1f} min, ~{est_total_seconds / 3600:.2f} hr)")

    if not skip_confirmation:
        resp = input("\nProceed with the full run? [y/N]: ").strip().lower()
        if resp != "y":
            print("Aborted. Re-run with a larger --stride to reduce this estimate, "
                  "or pass --yes to skip this prompt.")
            return

    # --- Real run, with incremental saving ---
    results = []
    start_time = time.time()

    for i, (row, col, lat, lon, h_target) in enumerate(candidate_points):
        mask = horizon_mask_for_point(dem, lat, lon, h_target,
                                       max_radius_km=max_radius_km,
                                       num_points=num_points)

        if np.all(np.isnan(mask)):
            continue  # entire radial scan was invalid, e.g. near grid edge

        worst_az = int(np.nanargmax(mask))
        best_az = int(np.nanargmin(mask))

        entry = {
            "row": row, "col": col, "lat": float(lat), "lon": float(lon),
            "h_target": float(h_target),
            "worst_azimuth": worst_az, "worst_angle_deg": float(mask[worst_az]),
            "best_azimuth": best_az, "best_angle_deg": float(mask[best_az]),
        }
        if save_full_profile:
            entry["full_profile_deg"] = mask.tolist()

        results.append(entry)

        if (i + 1) % checkpoint_every == 0 or (i + 1) == total_points:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total_points - (i + 1)) / rate if rate > 0 else float("nan")
            print(f"  {i + 1}/{total_points} done  "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

            with open(f"{output_basename}.json", "w") as f:
                json.dump({
                    "dem_basename": dem_basename,
                    "stride": stride,
                    "max_radius_km": max_radius_km,
                    "num_points": num_points,
                    "total_candidates": total_points,
                    "completed": i + 1,
                    "results": results,
                }, f)

    print(f"\nDone. Wrote {len(results)} results to {output_basename}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute horizon-mask summaries across the full LOLA grid (strided)."
    )
    parser.add_argument("dem_basename", help="Basename of the preprocessed DEM (e.g. ldem_1km)")
    parser.add_argument("output_basename", help="Basename for the output JSON results file")
    parser.add_argument("--stride", type=int, default=20,
                         help="Process every Nth row/col (default 20). Increase to "
                              "reduce runtime if the printed estimate is too large.")
    parser.add_argument("--max_radius_km", type=int, default=500,
                         help="Radial scan distance per point (default 500)")
    parser.add_argument("--num_points", type=int, default=360,
                         help="Number of azimuth directions per point (default 360)")
    parser.add_argument("--checkpoint_every", type=int, default=500,
                         help="Save partial results to disk every N points (default 500)")
    parser.add_argument("--save_full_profile", action="store_true",
                         help="Store the full 360-value profile per point, not just "
                              "worst/best summary. Only use with a large --stride - "
                              "this multiplies output size by ~num_points.")
    parser.add_argument("--yes", action="store_true", dest="skip_confirmation",
                         help="Skip the runtime-estimate confirmation prompt")

    args = parser.parse_args()

    run_full_grid(
        args.dem_basename, args.output_basename,
        stride=args.stride, max_radius_km=args.max_radius_km,
        num_points=args.num_points, checkpoint_every=args.checkpoint_every,
        save_full_profile=args.save_full_profile,
        skip_confirmation=args.skip_confirmation,
    )

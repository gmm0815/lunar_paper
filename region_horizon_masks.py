"""
region_horizon_masks.py

Computes the minimum-elevation-angle horizon mask (same HorizonScanner and radial
sampling schedule as full_grid_horizon_masks_fullres.py) for ONE latitude/longitude
box of the Moon, on a grid with ~0.5 km spacing, from the preprocessed 118 m LOLA
DEM, and writes everything - including the DEM elevation of every target - to a CSV.

Runs WITHOUT rasterio and without pandas: only numpy, plus
full_grid_horizon_masks_fullres.py in the same folder (it supplies the DEM reader and
the horizon scanner, so the calculation is identical to your other runs) and your
preprocessed DEM (e.g. ldem_118m_meta.json + its chunk files).

Defaults are the region you asked for:
    latitude -45 to -40 deg, longitude 10 to 15 deg, 0.5 km spacing, 500 km radius.

Usage:
    python region_horizon_masks.py
    python region_horizon_masks.py --dem ldem_118m --lat -45 -40 --lon 10 15 \\
        --spacing_km 0.5 --output region_m45_m40_10_15 --yes
    python region_horizon_masks.py --elevation_only          # DEM heights only, seconds
    python region_horizon_masks.py --resume --yes            # continue an interrupted run

Output (<output>.csv), one row per target:
    grid_row, grid_col   position in the target grid (grid_row 0 = southern edge,
                         grid_col 0 = western edge)
    row, col             DEM pixel the target falls in
    lat, lon             degrees
    h_target             DEM elevation (m above the 1,737.4 km reference sphere)
    worst_azimuth, worst_angle_deg   azimuth of, and value of, the HIGHEST horizon
    best_azimuth,  best_angle_deg    azimuth of, and value of, the LOWEST horizon
and <output>_meta.json with the settings used.

Grid options (--grid):
    exact (default)  0.5 km on the ground in BOTH directions: the longitude step is
                     recomputed for each latitude row (1/cos(lat)), so rows have slightly
                     different numbers of columns.
    rect             one regular lat/lon grid (every row has the same longitudes), with
                     the east-west step set for the box's middle latitude. East-west
                     spacing then runs from about 0.48 km at -45 deg to 0.52 km at -40 deg.
                     Easiest to reshape into an image: value[grid_row, grid_col].

Time: about 20 ms per target at 500 km (sandbox figure; your machine will differ) and
the 5 x 5 degree box has ~68,000 targets -> roughly 20-30 minutes. A timing probe is
printed first and asks for confirmation unless you pass --yes. The CSV is written as
it goes, so an interrupted run can be continued with --resume.
"""

import os
import sys
import json
import time
import argparse
import numpy as np

from full_grid_horizon_masks_fullres import (
    FullResDEM, HorizonScanner, build_distances_km, R_MOON_M,
)

KM_PER_DEG = 2.0 * np.pi * (R_MOON_M / 1000.0) / 360.0     # ~30.32 km per degree of arc
HEADER = ("grid_row,grid_col,row,col,lat,lon,h_target,"
          "worst_azimuth,worst_angle_deg,best_azimuth,best_angle_deg\n")


def build_grid(lat0, lat1, lon0, lon1, spacing_km, mode):
    """Target positions. Returns arrays grid_row, grid_col, lat, lon (flat, row-major from the SW corner)."""
    dlat = spacing_km / KM_PER_DEG
    n_lat = int(np.floor((lat1 - lat0) / dlat + 1e-9)) + 1
    lats = lat0 + dlat * np.arange(n_lat)
    gr, gc, la, lo = [], [], [], []
    mid_cos = np.cos(np.radians(0.5 * (lat0 + lat1)))
    for i, lat in enumerate(lats):
        c = mid_cos if mode == "rect" else np.cos(np.radians(lat))
        dlon = spacing_km / (KM_PER_DEG * c)
        n_lon = int(np.floor((lon1 - lon0) / dlon + 1e-9)) + 1
        lons = lon0 + dlon * np.arange(n_lon)
        gr.append(np.full(n_lon, i))
        gc.append(np.arange(n_lon))
        la.append(np.full(n_lon, lat))
        lo.append(lons)
    return (np.concatenate(gr), np.concatenate(gc), np.concatenate(la), np.concatenate(lo))


def fmt_az(a):
    return str(int(a)) if float(a).is_integer() else f"{a:g}"


def rows_already_done(path):
    """Number of complete data rows in an existing CSV; drops a half-written last line."""
    with open(path, "rb") as f:
        data = f.read()
    if not data.endswith(b"\n"):
        cut = data.rfind(b"\n")
        data = data[: cut + 1] if cut >= 0 else b""
        with open(path, "wb") as f:
            f.write(data)
    return max(0, data.count(b"\n") - 1)


def main():
    p = argparse.ArgumentParser(description="Horizon mask for one lat/lon box (numpy only).")
    p.add_argument("--dem", default="ldem_118m", help="preprocessed DEM basename (default ldem_118m)")
    p.add_argument("--lat", type=float, nargs=2, default=[-45.0, -40.0], metavar=("LAT0", "LAT1"))
    p.add_argument("--lon", type=float, nargs=2, default=[10.0, 15.0], metavar=("LON0", "LON1"))
    p.add_argument("--spacing_km", type=float, default=0.5)
    p.add_argument("--grid", choices=["exact", "rect"], default="exact")
    p.add_argument("--max_radius_km", type=float, default=500.0)
    p.add_argument("--num_azimuths", type=int, default=360)
    p.add_argument("--elevation_only", action="store_true",
                   help="Only write the DEM elevation at each target (no horizon calculation)")
    p.add_argument("--output", default="region_horizon", help="output basename (writes <output>.csv)")
    p.add_argument("--resume", action="store_true", help="continue an existing <output>.csv")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = p.parse_args()

    lat0, lat1 = sorted(args.lat)
    lon0, lon1 = sorted(args.lon)
    out_csv = f"{args.output}.csv"
    dem = FullResDEM(args.dem)
    pixel_km = dem.pixel_size_m / 1000.0
    print(f"DEM: {dem.n_rows} x {dem.n_cols} @ {dem.pixel_size_m:.2f} m/pixel")

    grid_row, grid_col, lat, lon = build_grid(lat0, lat1, lon0, lon1, args.spacing_km, args.grid)
    h = dem.sample_lonlat_deg(lon, lat)
    ok = np.isfinite(h)
    grid_row, grid_col, lat, lon, h = grid_row[ok], grid_col[ok], lat[ok], lon[ok], h[ok]
    total = len(h)
    pix_col = np.floor((R_MOON_M * np.radians(lon) - dem.c) / dem.a).astype(np.int64)
    pix_row = np.floor((R_MOON_M * np.radians(lat) - dem.f) / dem.e).astype(np.int64)
    if total == 0:
        raise SystemExit("No valid targets: the box has no DEM data (check --lat/--lon and the DEM).")
    print(f"Region: lat {lat0:g} to {lat1:g}, lon {lon0:g} to {lon1:g}; {args.spacing_km:g} km "
          f"spacing ({args.grid} grid) -> {total} targets with DEM data "
          f"({(~ok).sum()} positions had none)")
    print(f"  elevation range {h.min():.0f} to {h.max():.0f} m")

    scanner = None
    if not args.elevation_only:
        distances_km = build_distances_km(args.max_radius_km, pixel_km)
        scanner = HorizonScanner(dem, distances_km, args.num_azimuths)
        print(f"Radial samples per azimuth: {len(distances_km)} (first {distances_km[0] * 1000:.0f} m, "
              f"last {distances_km[-1]:.1f} km); {args.num_azimuths} azimuths; "
              f"max radius {args.max_radius_km:g} km")

        probe = np.unique(np.linspace(0, total - 1, min(8, total)).astype(int))
        t0 = time.time()
        for i in probe:
            scanner.profile(lat[i], lon[i], h[i])
        sec = (time.time() - t0) / len(probe)
        print(f"\nTiming: {sec:.3f} s/target over {len(probe)} spread-out targets "
              f"(first reads come off disk, so this tends to be pessimistic)")
        print(f"Estimated total: {sec * total / 3600:.2f} h  ({sec * total / 60:.0f} min)")

    start_at = 0
    if os.path.exists(out_csv):
        if args.resume:
            start_at = rows_already_done(out_csv)
            if start_at >= total:
                print(f"{out_csv} already has all {total} rows - nothing to do.")
                start_at = total
            else:
                print(f"Resuming {out_csv}: {start_at}/{total} rows already done.")
        elif not args.yes:
            raise SystemExit(f"{out_csv} already exists. Use --resume to continue it, --yes to overwrite, "
                             f"or a different --output.")
    if not args.yes and not args.elevation_only and start_at < total:
        if input("\nProceed with the full run? [y/N]: ").strip().lower() != "y":
            print("Aborted.")
            return

    meta = {"dem": args.dem, "lat": [lat0, lat1], "lon": [lon0, lon1], "spacing_km": args.spacing_km,
            "grid": args.grid, "max_radius_km": args.max_radius_km, "num_azimuths": args.num_azimuths,
            "elevation_only": args.elevation_only, "targets": int(total),
            "dem_resolution_m": dem.pixel_size_m}
    with open(f"{args.output}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    mode = "ab" if (args.resume and start_at > 0) else "wb"
    start = time.time()
    last_print = start
    try:
        with open(out_csv, mode) as f:
            if mode == "wb":
                f.write(HEADER.encode())
            for i in range(start_at, total):
                head = (f"{grid_row[i]},{grid_col[i]},{pix_row[i]},{pix_col[i]},"
                        f"{lat[i]:.8f},{lon[i]:.8f},{h[i]:.2f},")
                if scanner is None:
                    tail = ",,,"
                else:
                    prof = scanner.profile(lat[i], lon[i], h[i])
                    if np.all(np.isnan(prof)):
                        tail = ",,,"
                    else:
                        w, b = int(np.nanargmax(prof)), int(np.nanargmin(prof))
                        tail = (f"{fmt_az(scanner.azimuths_deg[w])},{prof[w]:.6f},"
                                f"{fmt_az(scanner.azimuths_deg[b])},{prof[b]:.6f}")
                f.write((head + tail + "\n").encode())

                done = i + 1
                if done % 500 == 0:
                    f.flush()
                now = time.time()
                if done % 1000 == 0 or done == total or now - last_print >= 60:
                    el = now - start
                    n_new = done - start_at
                    rate = n_new / el if el > 0 else 0
                    remain = (total - done) / rate / 60 if rate > 0 else float("nan")
                    print(f"  {done}/{total}  ({el / 60:.1f} min elapsed, ~{remain:.1f} min remaining)",
                          flush=True)
                    last_print = now
    except KeyboardInterrupt:
        print("\nInterrupted. Continue later with the same command plus --resume.")
        return

    print(f"\nWrote {out_csv}  ({os.path.getsize(out_csv) / 1e6:.2f} MB)")
    if scanner is not None:
        d = np.genfromtxt(out_csv, delimiter=",", names=True)
        for name in ("worst_angle_deg", "best_angle_deg"):
            v = d[name][np.isfinite(d[name])]
            if len(v):
                print(f"  {name}: min {v.min():.2f}  median {np.median(v):.2f}  mean {v.mean():.2f}  "
                      f"95th pct {np.percentile(v, 95):.2f}  max {v.max():.2f} deg")


if __name__ == "__main__":
    main()

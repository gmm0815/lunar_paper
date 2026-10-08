"""
region_horizon_masks.py   (1 km DEM version)

Computes the minimum-elevation-angle horizon mask for ONE latitude/longitude box of
the Moon, from the preprocessed ~1 km LOLA DEM (ldem_1km.npy + ldem_1km_meta.json, as
written by preprocess_lola_dem_downsampled.py), and writes everything - including the
DEM elevation of every target - to a CSV.

Needs ONLY numpy and lola_dem_reader.py (the same reader your other 1 km scripts use).
No rasterio, no pandas, no other project files.

The calculation is the same one as full_grid_horizon_masks.py / coordinates_to_min_el.py:
    angle = arctan(delta_h / d - d / (2 R))    maximised over distance (the horizon),
along 360 azimuths, with samples every --step_km (default 1 km, out to --max_radius_km,
default 500). Worst case = the highest horizon over all azimuths, best case = the lowest.
It is just vectorised (all 360 x 500 samples of a target in one pass) and reads the DEM
into memory once, so it runs much faster than the original per-azimuth loop.
Terrain is looked up exactly as lola_dem_reader.LunarDEM does (nearest pixel; no
longitude wrap; NaN outside the DEM or on nodata).

Defaults are the region you asked for:
    latitude -45 to -40 deg, longitude 10 to 15 deg, 0.5 km spacing, 500 km radius.

Usage (run it in the folder that holds the ldem_1km files):
    python region_horizon_masks.py
    python region_horizon_masks.py --dem ldem_1km --lat -45 -40 --lon 10 15 \\
        --spacing_km 0.5 --output region_m45_m40_10_15 --yes
    python region_horizon_masks.py --elevation_only          # DEM heights only, seconds
    python region_horizon_masks.py --resume --yes            # continue an interrupted run

IMPORTANT about resolution: the "1 km" DEM is really ~1.066 km per pixel (the 118 m
product decimated by 9). Targets spaced 0.5 km apart therefore fall on the SAME pixel
about four at a time (they get identical elevations, and mostly identical horizon
masks, with small differences from the changing viewpoint). That is allowed, but it adds
no new terrain detail and costs ~4x the time of one target per pixel. For one target per
DEM pixel use --spacing_km 1.066 (about 15,000 targets for this box).

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
    exact (default)  the requested spacing on the ground in BOTH directions: the longitude
                     step is recomputed for each latitude row (1/cos(lat)), so rows have
                     slightly different numbers of columns.
    rect             one regular lat/lon grid (every row has the same longitudes), with the
                     east-west step set for the box's middle latitude (east-west spacing runs
                     about 4% smaller at -45 deg and 4% larger at -40 deg). Easiest to
                     reshape into an image: value[grid_row, grid_col].

Time: ~5-20 ms per target at 500 km; the 5 x 5 degree box has ~68,000 targets at 0.5 km
-> roughly 6-25 minutes. A timing probe is printed first and asks for confirmation unless
you pass --yes. The CSV is written as it goes, so an interrupted run can be continued
with --resume.
"""

import os
import json
import time
import argparse
import numpy as np

from lola_dem_reader import LunarDEM, R_MOON_M

R_MOON_KM = R_MOON_M / 1000.0
KM_PER_DEG = 2.0 * np.pi * R_MOON_KM / 360.0               # ~30.32 km per degree of arc
HEADER = ("grid_row,grid_col,row,col,lat,lon,h_target,"
          "worst_azimuth,worst_angle_deg,best_azimuth,best_angle_deg\n")


# ----------------------------------------------------------------------------
# DEM held in memory (only the part a scan can touch)
# ----------------------------------------------------------------------------

def _colrow(dem, lon_rad, lat_rad):
    """Same mapping as LunarDEM.sample_many (nearest pixel), vectorised."""
    x = R_MOON_M * np.asarray(lon_rad, dtype=np.float64)
    y = R_MOON_M * np.asarray(lat_rad, dtype=np.float64)
    a, b, c, d, e, f = dem.a, dem.b, dem.c, dem.d, dem.e, dem.f
    det = dem._det
    cols = np.round((e * (x - c) - b * (y - f)) / det).astype(np.int64)
    rows = np.round((a * (y - f) - d * (x - c)) / det).astype(np.int64)
    return rows, cols


class WindowDEM:
    """The rows/cols of a LunarDEM that a scan from a lat/lon box can reach, in RAM."""

    def __init__(self, dem, lat_lo, lat_hi, lon_lo, lon_hi):
        self.dem = dem
        n_rows, n_cols = dem.shape
        r_a, c_a = _colrow(dem, np.radians([lon_lo, lon_hi]), np.radians([lat_lo, lat_lo]))
        r_b, c_b = _colrow(dem, np.radians([lon_lo, lon_hi]), np.radians([lat_hi, lat_hi]))
        pad = 3
        self.r0 = int(max(0, min(r_a.min(), r_b.min()) - pad))
        self.r1 = int(min(n_rows, max(r_a.max(), r_b.max()) + pad + 1))
        self.c0 = int(max(0, min(c_a.min(), c_b.min()) - pad))
        self.c1 = int(min(n_cols, max(c_a.max(), c_b.max()) + pad + 1))
        if self.r1 <= self.r0 or self.c1 <= self.c0:
            raise ValueError("Region lies outside the DEM")
        self.arr = self._read(self.r0, self.r1, self.c0, self.c1)

    def _read(self, r0, r1, c0, c1):
        dem = self.dem
        if not dem.chunked:
            return np.asarray(dem.arr[r0:r1, c0:c1], dtype=np.float64)
        out = np.full((r1 - r0, c1 - c0), np.nan)
        for k, ch in enumerate(dem.chunks):
            lo, hi = max(r0, ch["row_start"]), min(r1, ch["row_end"])
            if hi <= lo:
                continue
            arr = dem._get_chunk_array(k)
            out[lo - r0:hi - r0] = arr[lo - ch["row_start"]:hi - ch["row_start"], c0:c1]
        return out

    def sample(self, lon_rad, lat_rad):
        """Elevation (m) at lon/lat given in radians; NaN outside the DEM or on nodata."""
        rows, cols = _colrow(self.dem, lon_rad, lat_rad)
        n_rows, n_cols = self.dem.shape
        ok = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
        rr, cc = rows - self.r0, cols - self.c0
        inside = ok & (rr >= 0) & (rr < self.arr.shape[0]) & (cc >= 0) & (cc < self.arr.shape[1])
        out = np.full(np.shape(rows), np.nan)
        out[inside] = self.arr[rr[inside], cc[inside]]
        return out


def scan_window(lat0, lat1, lon0, lon1, max_radius_km):
    """lat/lon bounds (deg) that rays of length max_radius_km from anywhere in the box can reach."""
    big_d = max_radius_km / R_MOON_KM                       # central angle, radians
    lat_lo = max(-90.0, lat0 - np.degrees(big_d))
    lat_hi = min(90.0, lat1 + np.degrees(big_d))
    phi = np.radians(max(abs(lat0), abs(lat1)))
    if lat_lo <= -90.0 or lat_hi >= 90.0 or np.sin(big_d) >= np.cos(phi) or big_d >= np.pi / 2:
        return lat_lo, lat_hi, -180.0, 180.0                # may pass over a pole: take all longitudes
    dlon = np.degrees(np.arcsin(np.sin(big_d) / np.cos(phi))) + 1.0
    return lat_lo, lat_hi, lon0 - dlon, lon1 + dlon


# ----------------------------------------------------------------------------
# Horizon scan (same physics as full_grid_horizon_masks.py, vectorised)
# ----------------------------------------------------------------------------

class Scanner:
    def __init__(self, win, step_km, max_radius_km, num_azimuths=360):
        self.win = win
        self.distances_km = np.arange(step_km, max_radius_km + 1e-9, step_km)
        self.azimuths_deg = np.arange(num_azimuths) * (360.0 / num_azimuths)
        d_m = self.distances_km * 1000.0
        central = d_m / R_MOON_M
        bearing = np.radians(self.azimuths_deg)[:, None]
        sin_a, cos_a = np.sin(central)[None, :], np.cos(central)[None, :]
        self.cos_a = cos_a
        self.sa_cb = sin_a * np.cos(bearing)
        self.sb_sa = np.sin(bearing) * sin_a
        self.d_m = d_m[None, :]
        self.curvature_drop = (d_m ** 2 / (2.0 * R_MOON_M))[None, :]

    def profile(self, lat_deg, lon_deg, h_target):
        """Horizon angle (deg) per azimuth: max over distance of arctan(dh/d - d/2R)."""
        lat1, lon1 = np.radians(lat_deg), np.radians(lon_deg)
        sin1, cos1 = np.sin(lat1), np.cos(lat1)
        sin_lat2 = np.clip(sin1 * self.cos_a + cos1 * self.sa_cb, -1.0, 1.0)
        lat2 = np.arcsin(sin_lat2)
        lon2 = lon1 + np.arctan2(cos1 * self.sb_sa, self.cos_a - sin1 * sin_lat2)
        h = self.win.sample(lon2, lat2)
        slope = (h - h_target - self.curvature_drop) / self.d_m
        return np.degrees(np.arctan(np.fmax.reduce(slope, axis=1)))   # NaN-safe max


# ----------------------------------------------------------------------------
# Target grid
# ----------------------------------------------------------------------------

def build_grid(lat0, lat1, lon0, lon1, spacing_km, mode):
    """Target positions. Returns grid_row, grid_col, lat, lon (flat, row-major from the SW corner)."""
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
    p = argparse.ArgumentParser(description="Horizon mask for one lat/lon box from the 1 km DEM (numpy only).")
    p.add_argument("--dem", default="ldem_1km", help="preprocessed DEM basename (default ldem_1km)")
    p.add_argument("--lat", type=float, nargs=2, default=[-45.0, -40.0], metavar=("LAT0", "LAT1"))
    p.add_argument("--lon", type=float, nargs=2, default=[10.0, 15.0], metavar=("LON0", "LON1"))
    p.add_argument("--spacing_km", type=float, default=0.5)
    p.add_argument("--grid", choices=["exact", "rect"], default="exact")
    p.add_argument("--max_radius_km", type=float, default=500.0)
    p.add_argument("--step_km", type=float, default=1.0,
                   help="distance between samples along each ray (default 1 km, as in the original scripts)")
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

    dem = LunarDEM(args.dem)
    pixel_km = abs(dem.a) / 1000.0
    print(f"DEM: {dem.shape[0]} x {dem.shape[1]} @ {pixel_km * 1000:.0f} m/pixel "
          f"({'chunked' if dem.chunked else 'single file'})")
    if args.spacing_km < 0.9 * pixel_km:
        print(f"  Note: --spacing_km {args.spacing_km:g} is finer than the DEM's {pixel_km:.3f} km pixel, so "
              f"neighbouring targets share elevation values (~{(pixel_km / args.spacing_km) ** 2:.1f} targets per pixel).")

    reach = None
    if args.elevation_only:
        w_lat = (lat0 - 0.5, lat1 + 0.5, lon0 - 0.5, lon1 + 0.5)
    else:
        w_lat = scan_window(lat0, lat1, lon0, lon1, args.max_radius_km)
    t_load = time.time()
    win = WindowDEM(dem, *w_lat)
    print(f"Loaded DEM window: rows {win.r0}-{win.r1}, cols {win.c0}-{win.c1} "
          f"({win.arr.size * 8 / 1e6:.0f} MB in memory, {time.time() - t_load:.1f} s)")

    grid_row, grid_col, lat, lon = build_grid(lat0, lat1, lon0, lon1, args.spacing_km, args.grid)
    h = win.sample(np.radians(lon), np.radians(lat))
    ok = np.isfinite(h)
    grid_row, grid_col, lat, lon, h = grid_row[ok], grid_col[ok], lat[ok], lon[ok], h[ok]
    total = len(h)
    if total == 0:
        raise SystemExit("No valid targets: the box has no DEM data (check --lat/--lon and the DEM).")
    pix_row, pix_col = _colrow(dem, np.radians(lon), np.radians(lat))
    print(f"Region: lat {lat0:g} to {lat1:g}, lon {lon0:g} to {lon1:g}; {args.spacing_km:g} km "
          f"spacing ({args.grid} grid) -> {total} targets with DEM data ({(~ok).sum()} positions had none)")
    print(f"  elevation range {h.min():.0f} to {h.max():.0f} m")

    scanner = None
    if not args.elevation_only:
        scanner = Scanner(win, args.step_km, args.max_radius_km, args.num_azimuths)
        print(f"Samples per azimuth: {len(scanner.distances_km)} (every {args.step_km:g} km out to "
              f"{scanner.distances_km[-1]:g} km); {args.num_azimuths} azimuths")
        probe = np.unique(np.linspace(0, total - 1, min(8, total)).astype(int))
        t0 = time.time()
        for i in probe:
            scanner.profile(lat[i], lon[i], h[i])
        sec = (time.time() - t0) / len(probe)
        print(f"\nTiming: {sec:.3f} s/target over {len(probe)} spread-out targets")
        print(f"Estimated total: {sec * total / 3600:.2f} h  ({sec * total / 60:.0f} min)")

    start_at = 0
    if os.path.exists(out_csv):
        if args.resume:
            start_at = min(rows_already_done(out_csv), total)
            if start_at >= total:
                print(f"{out_csv} already has all {total} rows - nothing to do.")
            else:
                print(f"Resuming {out_csv}: {start_at}/{total} rows already done.")
        elif not args.yes:
            raise SystemExit(f"{out_csv} already exists. Use --resume to continue it, --yes to overwrite, "
                             f"or a different --output.")
    if not args.yes and not args.elevation_only and start_at < total:
        if input("\nProceed with the full run? [y/N]: ").strip().lower() != "y":
            print("Aborted.")
            return

    meta = {"dem": args.dem, "dem_pixel_km": pixel_km, "lat": [lat0, lat1], "lon": [lon0, lon1],
            "spacing_km": args.spacing_km, "grid": args.grid, "max_radius_km": args.max_radius_km,
            "step_km": args.step_km, "num_azimuths": args.num_azimuths,
            "elevation_only": args.elevation_only, "targets": int(total)}
    with open(f"{args.output}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    mode = "ab" if (args.resume and start_at > 0) else "wb"
    start = last_print = time.time()
    try:
        with open(out_csv, mode) as f:
            if mode == "wb":
                f.write(HEADER.encode())
            for i in range(start_at, total):
                head = (f"{grid_row[i]},{grid_col[i]},{pix_row[i]},{pix_col[i]},"
                        f"{lat[i]:.8f},{lon[i]:.8f},{h[i]:.2f},")
                tail = ",,,"
                if scanner is not None:
                    prof = scanner.profile(lat[i], lon[i], h[i])
                    if not np.all(np.isnan(prof)):
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
                    rate = (done - start_at) / el if el > 0 else 0
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

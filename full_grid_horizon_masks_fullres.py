"""
full_grid_horizon_masks_fullres.py

Horizon-mask scan that uses the FULL-resolution LOLA DEM (LDEM_256, ~118.45 m
per pixel) for the terrain, with targets on a coarse grid. Self-contained:
it does not import any of the other project files.

Three subcommands:

  preprocess  (run where rasterio is available)
      Converts the GeoTIFF into row-chunk .npy files holding the RAW int16
      values plus a small _meta.json. Scale/offset (0.5 / 0 for LDEM_256) are
      applied at read time, which keeps the data at ~8.5 GB instead of ~17 GB.

  scan        (needs only numpy - this is the one for the closed environment)
      Computes the horizon mask for targets every --stride full-res pixels.
      The output JSON uses the same schema as full_grid_horizon_masks.py, so
      json_to_csv.py and the plotting scripts work on it unchanged.

  check       (needs only numpy)
      Prints the elevation at a lat/lon, for spot-checking against known values.

Differences from the earlier 1 km pipeline (worth knowing when comparing runs):
  - Terrain is sampled from the 118 m grid, with a radial step that starts at
    one pixel and coarsens with distance (see RADIAL_SCHEDULE below).
  - Lookups use the pixel a point actually falls in (floor). The older reader
    used round(), which shifts lookups by up to one pixel.
  - Longitude wraps across +/-180 deg, so rays near the antimeridian keep their
    far-side samples (the older reader returned NaN for them).
  - Ray geometry is vectorised over all azimuths at once, and the maximum is
    taken on tan(angle) before a single arctan, which is mathematically the
    same result as taking arctan first.
  - Checkpoints are written to a temp file and swapped in with os.replace, so
    a copy of the JSON taken mid-run should not be half-written. On Windows the
    swap can fail if another program has the file open; the script then leaves
    the temp file and tells you.

Usage:
    python full_grid_horizon_masks_fullres.py preprocess LDEM_256_global.tif ldem_118m
    python full_grid_horizon_masks_fullres.py scan ldem_118m results_fullres_100km --stride 844
    python full_grid_horizon_masks_fullres.py check ldem_118m 19.53 -2.90

South-polar cap (targets on a regular km grid in a polar stereographic plane,
terrain still read from the global 118 m DEM):
    python full_grid_horizon_masks_fullres.py scan ldem_118m results_pole_1km \\
        --polar_spacing_km 1 --lat_max -80
Or keep the pixel-stride grid but only south of a latitude:
    python full_grid_horizon_masks_fullres.py scan ldem_118m results_cap --stride 8 --lat_max -80

Stride is in FULL-RES pixels (spacing ~= stride x 118.45 m):
    10 km -> 84     25 km -> 211    50 km -> 422
    100 km -> 844   500 km -> 4221  1000 km -> 8442
"""

import os
import sys
import json
import time
import argparse
import numpy as np

R_MOON_M = 1737400.0  # LOLA datum radius, same constant as the rest of the project

# Radial sampling schedule: (sample out to this distance in km, step in km).
# A step of None means "one native pixel". Fine steps near the target (where
# the 118 m data actually matters), coarser steps farther out, where terrain is
# increasingly hidden by curvature. If --max_radius_km exceeds the last entry,
# the last step size is reused out to the requested radius.
RADIAL_SCHEDULE = [
    (10.0, None),
    (50.0, 0.25),
    (150.0, 0.5),
    (300.0, 1.0),
    (500.0, 2.0),
]


def build_distances_km(max_radius_km, pixel_km, schedule=RADIAL_SCHEDULE):
    schedule = [(end, pixel_km if step is None else step) for end, step in schedule]
    if schedule[-1][0] < max_radius_km:
        schedule.append((max_radius_km, schedule[-1][1]))

    pieces = []
    pos = 0.0
    for end, step in schedule:
        end = min(end, max_radius_km)
        n = int(np.floor((end - pos) / step + 1e-9))
        if n <= 0:
            continue
        pieces.append(pos + step * np.arange(1, n + 1))
        pos += n * step
    return np.concatenate(pieces)


# ----------------------------------------------------------------------------
# Preprocessing (needs rasterio; imported lazily so the scan side doesn't)
# ----------------------------------------------------------------------------

def preprocess(input_tif, basename, max_chunk_mb=20.0):
    import rasterio
    from rasterio.windows import Window

    out_dir = os.path.dirname(basename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(basename)

    with rasterio.open(input_tif) as dem:
        n_rows, n_cols = dem.height, dem.width
        dtype = np.dtype(dem.dtypes[0])
        scale = dem.scales[0] if dem.scales and dem.scales[0] is not None else 1.0
        offset = dem.offsets[0] if dem.offsets and dem.offsets[0] is not None else 0.0
        nodata = dem.nodata
        t = dem.transform
        if t.b != 0 or t.d != 0:
            raise ValueError("Rotated/skewed transforms are not supported")

        # Decimal MB (1,000,000 bytes), to stay under GitHub's 25 MB limit
        row_bytes = n_cols * dtype.itemsize
        rows_per_chunk = max(1, int(max_chunk_mb * 1_000_000) // row_bytes)
        n_chunks = -(-n_rows // rows_per_chunk)
        total_gb = n_rows * row_bytes / 1e9

        print(f"Source: {n_rows} x {n_cols} {dtype}, pixel {abs(t.a):.2f} m, "
              f"scale={scale}, offset={offset}, nodata={nodata}")
        print(f"Writing {n_chunks} chunk files of {rows_per_chunk} rows "
              f"(~{rows_per_chunk * row_bytes / 1e6:.2f} MB each, {total_gb:.2f} GB total)")

        chunks = []
        for part in range(n_chunks):
            row_start = part * rows_per_chunk
            row_end = min(row_start + rows_per_chunk, n_rows)
            arr = dem.read(1, window=Window(0, row_start, n_cols, row_end - row_start))
            fname = f"{stem}_part{part:03d}.npy"
            np.save(os.path.join(out_dir, fname), arr)
            chunks.append({"file": fname, "row_start": row_start, "row_end": row_end})
            if part % 20 == 0 or part == n_chunks - 1:
                print(f"  chunk {part + 1}/{n_chunks} (rows {row_start}:{row_end})", flush=True)

        meta = {
            "format": "chunked_raw_v1",
            "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
            "shape": [n_rows, n_cols],
            "dtype": str(dtype),
            "scale": float(scale),
            "offset": float(offset),
            "nodata_dn": None if nodata is None else float(nodata),
            "pixel_size_m": float(abs(t.a)),
            "rows_per_chunk": rows_per_chunk,
            "chunks": chunks,
            "bounds": {"left": dem.bounds.left, "bottom": dem.bounds.bottom,
                       "right": dem.bounds.right, "top": dem.bounds.top},
            "crs_wkt": dem.crs.to_wkt() if dem.crs is not None else None,
            "source_file": input_tif,
        }

    with open(f"{basename}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {basename}_meta.json")


# ----------------------------------------------------------------------------
# Reader (numpy only)
# ----------------------------------------------------------------------------

class FullResDEM:
    """Chunked, memory-mapped reader for the raw-int16 layout written above."""

    def __init__(self, basename):
        meta_path = f"{basename}_meta.json"
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self._dir = os.path.dirname(os.path.abspath(meta_path))

        self.a, b, self.c, d, self.e, self.f = self.meta["transform"]
        if b != 0 or d != 0:
            raise ValueError("Rotated/skewed transforms are not supported")
        self.n_rows, self.n_cols = self.meta["shape"]
        self.rows_per_chunk = self.meta["rows_per_chunk"]
        self.chunks = self.meta["chunks"]
        self.scale = self.meta["scale"]
        self.offset = self.meta["offset"]
        self.nodata_dn = self.meta["nodata_dn"]
        self.pixel_size_m = self.meta["pixel_size_m"]
        self._cache = {}

        # Wrapping in longitude / clamping at the poles is only valid if the
        # grid really spans the whole sphere
        a_abs, e_abs = abs(self.a), abs(self.e)
        self.is_global = (
            abs(self.n_cols * a_abs - 2 * np.pi * R_MOON_M) < 2 * a_abs
            and abs(self.n_rows * e_abs - np.pi * R_MOON_M) < 2 * e_abs
        )
        if not self.is_global:
            print("Warning: DEM does not span the whole Moon; points outside it "
                  "will come back as NaN and there is no longitude wrap.")

        for k, ch in enumerate(self.chunks):
            if ch["row_start"] != k * self.rows_per_chunk:
                raise ValueError(f"Chunk {k} does not start at row {k * self.rows_per_chunk}")
        if self.chunks[-1]["row_end"] != self.n_rows:
            raise ValueError("Chunk manifest does not cover all rows")

    def _chunk(self, k):
        if k not in self._cache:
            path = os.path.join(self._dir, self.chunks[k]["file"])
            self._cache[k] = np.load(path, mmap_mode="r")
        return self._cache[k]

    def sample_xy(self, x, y):
        """Elevations (m) at projected coordinates x, y (metres, equirectangular
        on a sphere of radius R_MOON_M). Any array shape; NaN where invalid."""
        x = np.asarray(x, dtype=np.float64)
        shape = x.shape
        x = x.ravel()
        y = np.asarray(y, dtype=np.float64).ravel()

        cols = np.floor((x - self.c) / self.a).astype(np.int64)
        rows = np.floor((y - self.f) / self.e).astype(np.int64)

        if self.is_global:
            cols %= self.n_cols
            np.clip(rows, 0, self.n_rows - 1, out=rows)
            valid = None
        else:
            valid = ((rows >= 0) & (rows < self.n_rows)
                     & (cols >= 0) & (cols < self.n_cols))
            rows = np.where(valid, rows, 0)
            cols = np.where(valid, cols, 0)

        # Group lookups by chunk so each mmap'd file is indexed once
        chunk_idx = rows // self.rows_per_chunk
        order = np.argsort(chunk_idx, kind="stable")
        sorted_idx = chunk_idx[order]
        cuts = np.flatnonzero(np.diff(sorted_idx)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [len(order)]))

        raw = np.empty(rows.shape, dtype=np.dtype(self.meta["dtype"]))
        for s, t_end in zip(starts, ends):
            k = int(sorted_idx[s])
            sel = order[s:t_end]
            raw[sel] = self._chunk(k)[rows[sel] - k * self.rows_per_chunk, cols[sel]]

        out = raw.astype(np.float64) * self.scale + self.offset
        if self.nodata_dn is not None:
            out[raw == self.nodata_dn] = np.nan
        if valid is not None:
            out[~valid] = np.nan
        return out.reshape(shape)

    def sample_lonlat_deg(self, lon_deg, lat_deg):
        lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
        lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
        return self.sample_xy(R_MOON_M * lon, R_MOON_M * lat)

    def pixel_center_lonlat_deg(self, rows, cols):
        """lon/lat (deg) of pixel centres for arrays of row, col indices."""
        x = self.c + self.a * (np.asarray(cols) + 0.5)
        y = self.f + self.e * (np.asarray(rows) + 0.5)
        return np.degrees(x / R_MOON_M), np.degrees(y / R_MOON_M)


# ----------------------------------------------------------------------------
# Horizon scan
# ----------------------------------------------------------------------------

class HorizonScanner:
    """Worst-case (highest) horizon angle per azimuth for a target point.

    Same physics as before: angle = arctan(delta_h / d - d / (2R)) with
    delta_h the terrain height minus the target height, d the distance, and
    R the lunar radius. All azimuths and distances are evaluated in one
    vectorised pass, and trigonometry that depends only on the schedule is
    precomputed once.
    """

    def __init__(self, dem, distances_km, num_azimuths=360):
        self.dem = dem
        self.distances_km = np.asarray(distances_km, dtype=np.float64)
        self.azimuths_deg = np.arange(num_azimuths) * (360.0 / num_azimuths)

        d_m = self.distances_km * 1000.0
        central = d_m / R_MOON_M                         # central angle, radians
        bearing = np.radians(self.azimuths_deg)[:, None]
        sin_a, cos_a = np.sin(central)[None, :], np.cos(central)[None, :]

        self.cos_a = cos_a
        self.sa_cb = sin_a * np.cos(bearing)
        self.sb_sa = np.sin(bearing) * sin_a
        self.d_m = d_m[None, :]
        self.curvature_drop = (d_m ** 2 / (2.0 * R_MOON_M))[None, :]

    def profile(self, lat_deg, lon_deg, h_target):
        """Worst-case horizon angle in degrees, one value per azimuth (NaN if
        a whole ray had no valid terrain)."""
        lat1 = np.radians(lat_deg)
        lon1 = np.radians(lon_deg)
        sin1, cos1 = np.sin(lat1), np.cos(lat1)

        sin_lat2 = np.clip(sin1 * self.cos_a + cos1 * self.sa_cb, -1.0, 1.0)
        lat2 = np.arcsin(sin_lat2)
        lon2 = lon1 + np.arctan2(cos1 * self.sb_sa, self.cos_a - sin1 * sin_lat2)

        h = self.dem.sample_xy(R_MOON_M * lon2, R_MOON_M * lat2)

        # arctan is monotonic, so the max angle is the arctan of the max slope
        slope = (h - h_target - self.curvature_drop) / self.d_m
        worst = np.fmax.reduce(slope, axis=1)  # ignores NaN, no all-NaN warning
        return np.degrees(np.arctan(worst))


def write_json_atomic(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    try:
        os.replace(tmp, path)
        return True
    except PermissionError:
        print(f"  Could not replace {path} (open in another program?). "
              f"Latest results are in {tmp}")
        return False


def south_polar_cap_grid(spacing_km, lat_max_deg=None, region=None):
    """Regular x/y grid (km) in a south-polar stereographic plane (true scale at
    the pole, sphere radius R_MOON_M). Grid lines sit on multiples of spacing_km.
    Either the whole cap south of lat_max_deg, or a square region
    (centre_x_km, centre_y_km, width_km), optionally also limited to lat_max_deg.
    Returns x_km, y_km, lat_deg, lon_deg as flat arrays.

        rho = 2 R tan((90 deg + lat) / 2)      x = rho sin(lon)    y = rho cos(lon)
    """
    r_km = R_MOON_M / 1000.0
    if region is None:
        rho_cap_km = 2.0 * r_km * np.tan(np.radians(90.0 + lat_max_deg) / 2.0)
        n = int(np.floor(rho_cap_km / spacing_km))
        xs = ys = np.arange(-n, n + 1) * spacing_km    # includes the pole itself
    else:
        cx, cy, width = region
        xs = np.arange(np.ceil((cx - width / 2) / spacing_km),
                       np.floor((cx + width / 2) / spacing_km) + 1) * spacing_km
        ys = np.arange(np.ceil((cy - width / 2) / spacing_km),
                       np.floor((cy + width / 2) / spacing_km) + 1) * spacing_km
    xg, yg = np.meshgrid(xs, ys, indexing="xy")
    x, y = xg.ravel(), yg.ravel()
    rho = np.hypot(x, y)
    if lat_max_deg is not None:
        rho_cap_km = 2.0 * r_km * np.tan(np.radians(90.0 + lat_max_deg) / 2.0)
        inside = rho <= rho_cap_km + 1e-9
        x, y, rho = x[inside], y[inside], rho[inside]
    lat = -90.0 + 2.0 * np.degrees(np.arctan(rho / (2.0 * R_MOON_M / 1000.0)))
    lon = np.degrees(np.arctan2(x, y))                # 0 deg at +y, east toward +x
    return x, y, lat, lon


def run_scan(dem_basename, output_basename, stride=None, max_radius_km=500.0,
             num_azimuths=360, checkpoint_seconds=300, skip_confirmation=False,
             n_timing_samples=8, lat_max_deg=None, polar_spacing_km=None, region=None):
    dem = FullResDEM(dem_basename)
    pixel_km = dem.pixel_size_m / 1000.0
    distances_km = build_distances_km(max_radius_km, pixel_km)
    scanner = HorizonScanner(dem, distances_km, num_azimuths)

    print(f"DEM: {dem.n_rows} x {dem.n_cols} @ {dem.pixel_size_m:.2f} m/pixel")

    xy_t = None
    if polar_spacing_km is not None:
        # --- Regular grid in the south-polar stereographic plane ---
        if lat_max_deg is None and region is None:
            lat_max_deg = -80.0
        x_flat, y_flat, lat_flat, lon_flat = south_polar_cap_grid(
            polar_spacing_km, lat_max_deg, region)
        h_flat = dem.sample_lonlat_deg(lon_flat, lat_flat)
        keep = ~np.isnan(h_flat)
        lats_t, lons_t, h_t = lat_flat[keep], lon_flat[keep], h_flat[keep]
        xy_t = (x_flat[keep], y_flat[keep])
        # equirectangular pixel each target falls in (kept for schema compatibility)
        cols_t = np.floor((R_MOON_M * np.radians(lons_t) - dem.c) / dem.a).astype(np.int64)
        rows_t = np.floor((R_MOON_M * np.radians(lats_t) - dem.f) / dem.e).astype(np.int64)
        total = len(h_t)
        spacing_km = polar_spacing_km
        where = (f"region centre ({region[0]:g}, {region[1]:g}) km, width {region[2]:g} km"
                 if region is not None else f"south of {lat_max_deg:g} deg")
        print(f"Targets: south-polar grid, {polar_spacing_km:g} km spacing, {where} "
              f"-> {len(h_flat)} positions, {total} valid")
    else:
        # --- Target grid on the full-resolution DEM, one target per stride pixels ---
        rows_axis = np.arange(0, dem.n_rows, stride)
        if lat_max_deg is not None and dem.e < 0:
            # only build positions in the requested latitude band (a small stride
            # over the whole globe would otherwise exhaust memory before filtering)
            first_row = (R_MOON_M * np.radians(lat_max_deg) - dem.f) / dem.e - 0.5
            rows_axis = rows_axis[rows_axis >= int(np.ceil(first_row))]
        cols_axis = np.arange(0, dem.n_cols, stride)
        row_grid, col_grid = np.meshgrid(rows_axis, cols_axis, indexing="ij")
        row_flat, col_flat = row_grid.ravel(), col_grid.ravel()
        lon_flat, lat_flat = dem.pixel_center_lonlat_deg(row_flat, col_flat)
        h_flat = dem.sample_lonlat_deg(lon_flat, lat_flat)

        keep = ~np.isnan(h_flat)
        if lat_max_deg is not None:
            keep &= lat_flat <= lat_max_deg
        rows_t, cols_t = row_flat[keep], col_flat[keep]
        lats_t, lons_t, h_t = lat_flat[keep], lon_flat[keep], h_flat[keep]
        total = len(h_t)

        spacing_km = stride * pixel_km
        cap_note = f", south of {lat_max_deg:g} deg only" if lat_max_deg is not None else ""
        print(f"Targets: stride {stride} px (~{spacing_km:.1f} km){cap_note} -> "
              f"{len(rows_axis)} x {len(cols_axis)} = {len(row_flat)} positions, "
              f"{total} valid")
    print(f"Radial samples per azimuth: {len(distances_km)} "
          f"(first {distances_km[0] * 1000:.0f} m, last {distances_km[-1]:.1f} km); "
          f"{num_azimuths} azimuths")
    if total == 0:
        print("No valid targets - check --stride and the DEM.")
        return

    # --- Time a spread of targets, extrapolate, confirm ---
    probe = np.unique(np.linspace(0, total - 1, min(n_timing_samples, total)).astype(int))
    t0 = time.time()
    for i in probe:
        scanner.profile(lats_t[i], lons_t[i], h_t[i])
    sec_per_point = (time.time() - t0) / len(probe)
    est = sec_per_point * total
    print(f"\nTiming: {sec_per_point:.3f} s/point over {len(probe)} spread-out targets "
          f"(first reads come off disk, so this tends to be pessimistic)")
    print(f"Estimated total: {est / 3600:.2f} h  ({est / 60:.0f} min)")

    if not skip_confirmation:
        if input("\nProceed with the full run? [y/N]: ").strip().lower() != "y":
            print("Aborted. Use a larger --stride or smaller --max_radius_km to cut this, "
                  "or --yes to skip the prompt.")
            return

    out_path = f"{output_basename}.json"
    results = []
    start = last_ckpt = time.time()
    completed = 0

    def payload():
        return {
            "dem_basename": dem_basename,
            "stride": stride,
            "stride_unit": ("south-polar grid km" if xy_t is not None
                            else "full-res pixels"),
            "projection": ("south_polar_stereographic" if xy_t is not None
                           else "equirectangular"),
            "lat_max_deg": lat_max_deg,
            "region_km": list(region) if region is not None else None,
            "point_spacing_km": spacing_km,
            "dem_resolution_m": dem.pixel_size_m,
            "max_radius_km": max_radius_km,
            "num_points": num_azimuths,
            "radial_samples_per_azimuth": int(len(distances_km)),
            "total_candidates": total,
            "completed": completed,
            "results": results,
        }

    try:
        for i in range(total):
            prof = scanner.profile(lats_t[i], lons_t[i], h_t[i])
            completed = i + 1

            if not np.all(np.isnan(prof)):
                w, b = int(np.nanargmax(prof)), int(np.nanargmin(prof))
                az_w, az_b = scanner.azimuths_deg[w], scanner.azimuths_deg[b]
                results.append({
                    "row": int(rows_t[i]), "col": int(cols_t[i]),
                    "lat": float(lats_t[i]), "lon": float(lons_t[i]),
                    "h_target": float(h_t[i]),
                    **({"x_km": float(xy_t[0][i]), "y_km": float(xy_t[1][i])}
                       if xy_t is not None else {}),
                    "worst_azimuth": int(az_w) if float(az_w).is_integer() else float(az_w),
                    "worst_angle_deg": float(prof[w]),
                    "best_azimuth": int(az_b) if float(az_b).is_integer() else float(az_b),
                    "best_angle_deg": float(prof[b]),
                })

            if completed % 1000 == 0 or completed == total:
                el = time.time() - start
                rate = completed / el
                print(f"  {completed}/{total}  ({el / 60:.1f} min elapsed, "
                      f"~{(total - completed) / rate / 60:.1f} min remaining)", flush=True)

            if time.time() - last_ckpt >= checkpoint_seconds:
                write_json_atomic(out_path, payload())
                last_ckpt = time.time()
    except KeyboardInterrupt:
        print(f"\nInterrupted after {completed}/{total}; saving what was computed.")

    write_json_atomic(out_path, payload())
    print(f"\nWrote {len(results)} results ({completed}/{total} targets processed) to {out_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess", help="GeoTIFF -> chunked raw .npy + meta JSON (needs rasterio)")
    p.add_argument("input_tif")
    p.add_argument("basename", help="Output basename, e.g. ldem_118m")
    p.add_argument("--max_chunk_mb", type=float, default=20.0,
                   help="Max size per chunk in DECIMAL MB (default 20)")

    p = sub.add_parser("scan", help="Run the horizon-mask scan (numpy only)")
    p.add_argument("dem_basename")
    p.add_argument("output_basename")
    p.add_argument("--stride", type=int, default=None,
                   help="Target spacing in FULL-RES pixels (~118.45 m each)")
    p.add_argument("--lat_max", type=float, default=None,
                   help="Only scan targets at or south of this latitude, e.g. -80. "
                        "Works with --stride; with --polar_spacing_km it sets the cap "
                        "edge (default -80).")
    p.add_argument("--region", type=float, nargs=3, metavar=("CX_KM", "CY_KM", "WIDTH_KM"),
                   default=None,
                   help="With --polar_spacing_km: scan only this square (polar-plane km) "
                        "instead of the whole cap")
    p.add_argument("--polar_spacing_km", type=float, default=None,
                   help="Instead of --stride, place targets on a regular x/y grid (km) "
                        "in a south-polar stereographic plane, out to --lat_max. "
                        "Best choice for polar maps/overlays.")
    p.add_argument("--max_radius_km", type=float, default=500.0)
    p.add_argument("--num_azimuths", type=int, default=360)
    p.add_argument("--checkpoint_seconds", type=float, default=300.0,
                   help="Write partial results this often (default 300 s)")
    p.add_argument("--yes", action="store_true", dest="skip_confirmation",
                   help="Skip the runtime-estimate prompt")

    p = sub.add_parser("check", help="Print the elevation at a lat/lon")
    p.add_argument("dem_basename")
    p.add_argument("lat", type=float)
    p.add_argument("lon", type=float)

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess(args.input_tif, args.basename, args.max_chunk_mb)
    elif args.command == "scan":
        if (args.stride is None) == (args.polar_spacing_km is None):
            parser.error("scan needs exactly one of --stride or --polar_spacing_km")
        if args.region is not None and args.polar_spacing_km is None:
            parser.error("--region needs --polar_spacing_km")
        run_scan(args.dem_basename, args.output_basename, args.stride,
                 max_radius_km=args.max_radius_km, num_azimuths=args.num_azimuths,
                 checkpoint_seconds=args.checkpoint_seconds,
                 skip_confirmation=args.skip_confirmation,
                 lat_max_deg=args.lat_max, polar_spacing_km=args.polar_spacing_km,
                 region=args.region)
    elif args.command == "check":
        dem = FullResDEM(args.dem_basename)
        h = float(dem.sample_lonlat_deg(args.lon, args.lat))
        print(f"DEM {dem.n_rows} x {dem.n_cols} @ {dem.pixel_size_m:.2f} m/pixel, "
              f"{len(dem.chunks)} chunks")
        print(f"lat={args.lat}, lon={args.lon}: "
              + ("elevation = %.1f m" % h if not np.isnan(h) else "no data"))


if __name__ == "__main__":
    main()

"""
preprocess_lola_dem_downsampled.py

Run this ONCE, on a machine with rasterio installed, to convert the FULL
global LOLA GeoTIFF DEM into a DOWNSAMPLED plain .npy array + JSON
metadata sidecar.

Rather than reading every pixel, this decimates by a fixed factor (e.g.
factor=9 turns 118.45m/pixel into ~1066m/pixel) by requesting a smaller
`out_shape` from rasterio. GDAL performs this decimation efficiently at
the block level, so the full-resolution array is never materialized in
Python memory - the output array is the small, already-downsampled size
from the start. For factor=9 the global 118m product shrinks to roughly
52 million pixels (~210MB as float32), easily fitting in RAM with no
windowing or chunking required.

Resampling method matters for horizon-obstruction work specifically:
  - "nearest"  : picks one literal source pixel per output cell (true
                 "every Nth coordinate" sampling, matches what you asked for)
  - "max"      : takes the max value in each downsampled block - more
                 conservative for horizon/obstruction calculations, since
                 it won't accidentally skip over a narrow peak that a
                 nearest-neighbor sample happened to miss
Default is "nearest" to match your request; pass --resampling max if you
want the more conservative worst-case-obstruction behavior instead.

Usage:
    python preprocess_lola_dem_downsampled.py Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif ldem_1km
    python preprocess_lola_dem_downsampled.py <tif> <basename> --factor 9 --resampling nearest
    python preprocess_lola_dem_downsampled.py <tif> <basename> --factor 9 --resampling max
"""

import sys
import json
import argparse
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from affine import Affine

# NOTE: "max"/"min"/"mode"/"med"/"q1"/"q3" are warp-only resampling methods
# in GDAL - they raise ResamplingAlgorithmError on a plain dataset.read()
# with out_shape. Routing everything through a WarpedVRT works for ALL of
# these methods uniformly (including "nearest"/"average"), and GDAL's warp
# engine still processes internally in blocks, so this does not reintroduce
# the full-array-in-memory problem.
RESAMPLING_MAP = {
    "nearest": Resampling.nearest,
    "max": Resampling.max,
    "average": Resampling.average,
}

# Target max size per output .npy chunk, so files stay uploadable through
# GitHub's web UI (25MB hard limit) without needing Git LFS.
#
# IMPORTANT: GitHub's 25MB limit is decimal megabytes (1,000,000 bytes), not
# binary mebibytes (1,048,576 bytes). Using 1024*1024 here under-shoots the
# real limit - 24.5 "MiB" is actually ~25.69 decimal MB, which is OVER the
# cap. Windows Explorer compounds the confusion by displaying sizes in KiB
# but labeling the column "KB".
#
# Using decimal MB here, with a lower default (20MB) for real headroom
# against rounding, the .npy header (~128 bytes, negligible but nonzero),
# and any filesystem/transfer overhead.
GITHUB_MAX_BYTES = int(20 * 1_000_000)


def preprocess_dem_downsampled(input_tif_path, output_basename, factor=9,
                                resampling="nearest", max_chunk_mb=20):
    resampling_method = RESAMPLING_MAP[resampling]
    max_chunk_bytes = int(max_chunk_mb * 1_000_000)  # decimal MB, not MiB

    with rasterio.open(input_tif_path) as dem:
        n_rows, n_cols = dem.height, dem.width
        new_rows = max(1, n_rows // factor)
        new_cols = max(1, n_cols // factor)

        est_size_mb = (new_rows * new_cols * 4) / (1024 ** 2)  # float32
        native_res = dem.res[0]
        new_res = native_res * factor

        print(f"Source: {n_rows} x {n_cols} pixels @ {native_res:.2f} m/pixel")
        print(f"Decimation factor: {factor} -> {new_rows} x {new_cols} @ ~{new_res:.1f} m/pixel")
        print(f"Estimated output size: {est_size_mb:.1f} MB")
        print(f"Resampling method: {resampling}")

        # --- Build the downsampled transform up front (same CRS, coarser grid) ---
        scale_x = n_cols / new_cols
        scale_y = n_rows / new_rows
        new_transform = dem.transform * Affine.scale(scale_x, scale_y)

        # --- Warp-based decimated read: works for nearest/average/max/min/etc,
        #     and GDAL still processes this in blocks internally, so memory
        #     stays proportional to the OUTPUT size, not the source size ---
        with WarpedVRT(
            dem,
            crs=dem.crs,
            transform=new_transform,
            width=new_cols,
            height=new_rows,
            resampling=resampling_method,
        ) as vrt:
            raw = vrt.read(1)

        scale = dem.scales[0] if dem.scales and dem.scales[0] is not None else 1.0
        offset = dem.offsets[0] if dem.offsets and dem.offsets[0] is not None else 0.0
        nodata = dem.nodata

        arr_elevation = raw.astype(np.float32) * scale + offset
        if nodata is not None:
            arr_elevation[raw == nodata] = np.nan

        # new_transform was already computed above and used for the VRT read -
        # reuse it directly so the saved metadata matches exactly what was read
        t = new_transform
        transform = [t.a, t.b, t.c, t.d, t.e, t.f]

        crs_wkt = dem.crs.to_wkt() if dem.crs is not None else None
        bounds = {
            "left": dem.bounds.left, "bottom": dem.bounds.bottom,
            "right": dem.bounds.right, "top": dem.bounds.top,
        }

    # --- Split into row-chunks so no single file exceeds GITHUB_MAX_BYTES ---
    # (default 24.5MB, comfortably under GitHub's 25MB web-upload limit,
    # avoiding the need for Git LFS)
    itemsize = arr_elevation.dtype.itemsize  # 4 bytes for float32
    row_bytes = new_cols * itemsize
    rows_per_chunk = max(1, max_chunk_bytes // row_bytes)

    chunk_manifest = []
    part_idx = 0
    for row_start in range(0, new_rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, new_rows)
        chunk_arr = arr_elevation[row_start:row_end, :]

        chunk_filename = f"{output_basename}_part{part_idx:03d}.npy"
        np.save(chunk_filename, chunk_arr)

        chunk_bytes = chunk_arr.nbytes
        chunk_manifest.append({
            "file": chunk_filename,
            "row_start": row_start,
            "row_end": row_end,
        })
        print(f"  wrote {chunk_filename}  rows [{row_start}:{row_end})  "
              f"({chunk_bytes / 1_000_000:.2f} decimal MB, {chunk_bytes} bytes)")
        part_idx += 1

    meta = {
        "transform": transform,
        "shape": list(arr_elevation.shape),
        "dtype": str(arr_elevation.dtype),
        "nodata": "nan",
        "bounds": bounds,
        "crs_wkt": crs_wkt,
        "source_file": input_tif_path,
        "scale_applied": float(scale),
        "offset_applied": float(offset),
        "decimation_factor": factor,
        "approx_resolution_m": float(new_res),
        "resampling_method": resampling,
        "rows_per_chunk": rows_per_chunk,   # constant for all but possibly the last chunk
        "chunks": chunk_manifest,
    }
    with open(f"{output_basename}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {len(chunk_manifest)} chunk files + {output_basename}_meta.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample and preprocess a global LOLA DEM (low memory, no windowing needed)."
    )
    parser.add_argument("input_tif", help="Path to the source GeoTIFF")
    parser.add_argument("output_basename", help="Basename for .npy/.json outputs")
    parser.add_argument("--factor", type=int, default=9,
                         help="Decimation factor - every Nth pixel (default 9, "
                              "~118m -> ~1066m at the LOLA 118m product's native res)")
    parser.add_argument("--resampling", choices=list(RESAMPLING_MAP.keys()),
                         default="nearest",
                         help="Resampling method (default: nearest = literal every-Nth-pixel sampling)")
    parser.add_argument("--max_chunk_mb", type=float, default=20,
                         help="Max size per output .npy chunk in DECIMAL MB "
                              "(1,000,000 bytes; default 20, for real headroom "
                              "under GitHub's 25MB decimal-MB web-upload limit)")

    args = parser.parse_args()

    preprocess_dem_downsampled(
        args.input_tif, args.output_basename,
        factor=args.factor, resampling=args.resampling,
        max_chunk_mb=args.max_chunk_mb,
    )

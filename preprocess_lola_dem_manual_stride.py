"""
preprocess_lola_dem_manual_stride.py

Run this ONCE, on a machine with rasterio installed. Produces the same
kind of downsampled output as preprocess_lola_dem_downsampled.py, but
does the "every Nth pixel" selection explicitly in Python/numpy rather
than relying on GDAL's out_shape/WarpedVRT decimation logic. Useful if
you want to be 100% certain exactly which source pixel gets kept.

Still memory-safe: reads the source raster in row-chunks (never the
full array at once), and within each chunk keeps only every `factor`-th
row and column before writing to the output.

Usage:
    python preprocess_lola_dem_manual_stride.py Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif ldem_1km --factor 9
"""

import sys
import json
import argparse
import numpy as np
import rasterio


def preprocess_dem_manual_stride(input_tif_path, output_basename, factor=9,
                                  chunk_rows=4000):
    with rasterio.open(input_tif_path) as dem:
        n_rows, n_cols = dem.height, dem.width
        scale = dem.scales[0] if dem.scales and dem.scales[0] is not None else 1.0
        offset = dem.offsets[0] if dem.offsets and dem.offsets[0] is not None else 0.0
        nodata = dem.nodata

        new_rows = len(range(0, n_rows, factor))
        new_cols = len(range(0, n_cols, factor))
        native_res = dem.res[0]
        new_res = native_res * factor

        print(f"Source: {n_rows} x {n_cols} @ {native_res:.2f} m/pixel")
        print(f"Stride factor: {factor} -> {new_rows} x {new_cols} @ ~{new_res:.1f} m/pixel")

        out_arr = np.empty((new_rows, new_cols), dtype=np.float32)

        out_row = 0
        # Only need to actually READ rows that will be kept, but we still
        # chunk the read so no more than `chunk_rows` worth of ORIGINAL rows
        # are ever in memory at once.
        for row_start in range(0, n_rows, chunk_rows):
            row_end = min(row_start + chunk_rows, n_rows)
            window = rasterio.windows.Window(
                col_off=0, row_off=row_start,
                width=n_cols, height=row_end - row_start
            )
            raw_chunk = dem.read(1, window=window)  # only this chunk in RAM

            # Which rows within this chunk fall on the stride grid?
            # (global row index r is kept iff r % factor == 0)
            first_kept = (-row_start) % factor  # offset within chunk of first kept row
            kept_rows = raw_chunk[first_kept::factor, ::factor]

            n_kept = kept_rows.shape[0]
            chunk_out = kept_rows.astype(np.float32) * scale + offset
            if nodata is not None:
                chunk_out[kept_rows == nodata] = np.nan

            out_arr[out_row:out_row + n_kept, :] = chunk_out
            out_row += n_kept

            pct = 100.0 * row_end / n_rows
            print(f"  processed source rows {row_start}:{row_end}  ({pct:.1f}%)", flush=True)

        assert out_row == new_rows, f"expected {new_rows} output rows, got {out_row}"

        # --- Transform for the strided grid: same origin, scaled pixel size ---
        t = dem.transform
        new_transform = [
            t.a * factor, t.b, t.c,
            t.d, t.e * factor, t.f,
        ]

        crs_wkt = dem.crs.to_wkt() if dem.crs is not None else None
        bounds = {
            "left": dem.bounds.left, "bottom": dem.bounds.bottom,
            "right": dem.bounds.right, "top": dem.bounds.top,
        }

    np.save(f"{output_basename}.npy", out_arr)

    meta = {
        "transform": new_transform,
        "shape": [new_rows, new_cols],
        "dtype": str(out_arr.dtype),
        "nodata": "nan",
        "bounds": bounds,
        "crs_wkt": crs_wkt,
        "source_file": input_tif_path,
        "scale_applied": float(scale),
        "offset_applied": float(offset),
        "decimation_factor": factor,
        "approx_resolution_m": float(new_res),
        "resampling_method": "manual_stride_nearest",
    }
    with open(f"{output_basename}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {output_basename}.npy  shape={out_arr.shape} dtype={out_arr.dtype}")
    print(f"Wrote {output_basename}_meta.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manual every-Nth-pixel decimation of a global LOLA DEM (low memory, no GDAL resampling)."
    )
    parser.add_argument("input_tif", help="Path to the source GeoTIFF")
    parser.add_argument("output_basename", help="Basename for .npy/.json outputs")
    parser.add_argument("--factor", type=int, default=9,
                         help="Keep every Nth row/column (default 9)")
    parser.add_argument("--chunk_rows", type=int, default=4000,
                         help="Source rows read per chunk (default 4000)")

    args = parser.parse_args()

    preprocess_dem_manual_stride(
        args.input_tif, args.output_basename,
        factor=args.factor, chunk_rows=args.chunk_rows,
    )

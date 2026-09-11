"""
preprocess_lola_dem.py

Run this ONCE, on a machine with rasterio installed, to convert a LOLA
GeoTIFF DEM into a plain .npy array + a JSON metadata sidecar. The output
can then be loaded in a closed environment using only numpy + stdlib json.

Handles the LOLA scale/offset convention automatically (e.g. the 118m
global product is stored as int16 with scale=0.5), converting to real
elevation-in-meters as float32 before saving, and preserves nodata as NaN.

Usage:
    python preprocess_lola_dem.py Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif ldem_118m
"""

import sys
import json
import numpy as np
import rasterio


def preprocess_dem(input_tif_path, output_basename):
    with rasterio.open(input_tif_path) as dem:
        # --- Read the full elevation array (band 1, assumes single-band raster) ---
        raw = dem.read(1)

        # --- Apply LOLA's internal scale/offset convention, if present ---
        # e.g. the 118m global product stores int16 DN with scale=0.5, offset=0.0
        # so real elevation (m) = DN * scale + offset
        scale = dem.scales[0] if dem.scales and dem.scales[0] is not None else 1.0
        offset = dem.offsets[0] if dem.offsets and dem.offsets[0] is not None else 0.0

        nodata = dem.nodata

        arr_elevation = raw.astype(np.float32) * scale + offset

        # Preserve nodata as NaN (must be done on the raw DN, before/alongside
        # the scale math, so the sentinel value doesn't get silently corrupted)
        if nodata is not None:
            arr_elevation[raw == nodata] = np.nan

        # --- Affine transform: (col, row) -> (x, y) in the raster's CRS ---
        t = dem.transform
        transform = [t.a, t.b, t.c, t.d, t.e, t.f]

        crs_wkt = dem.crs.to_wkt() if dem.crs is not None else None
        shape = arr_elevation.shape

        bounds = {
            "left": dem.bounds.left,
            "bottom": dem.bounds.bottom,
            "right": dem.bounds.right,
            "top": dem.bounds.top,
        }

    # --- Save the elevation array (already in real meters, NaN = nodata) ---
    np.save(f"{output_basename}.npy", arr_elevation)

    # --- Save metadata needed to do lon/lat -> row/col math at runtime ---
    meta = {
        "transform": transform,       # [a, b, c, d, e, f]
        "shape": shape,                # (rows, cols)
        "dtype": str(arr_elevation.dtype),
        "nodata": "nan",               # nodata is now represented as NaN
        "bounds": bounds,
        "crs_wkt": crs_wkt,
        "source_file": input_tif_path,
        "scale_applied": float(scale),
        "offset_applied": float(offset),
    }
    with open(f"{output_basename}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {output_basename}.npy  shape={shape} dtype={arr_elevation.dtype}")
    print(f"Wrote {output_basename}_meta.json")
    print(f"Applied scale={scale}, offset={offset}")
    print(f"CRS: {(crs_wkt[:80] + '...') if crs_wkt else 'None'}")
    print(f"Bounds: {bounds}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_lola_dem.py <input.tif> <output_basename>")
        sys.exit(1)

    preprocess_dem(sys.argv[1], sys.argv[2])

"""
lola_dem_reader.py

Runtime DEM reader for closed environments - depends only on numpy and
the standard library.

Supports two storage layouts, chosen automatically based on what the
metadata JSON contains:

  1. Single-file:  "{basename}.npy" + "{basename}_meta.json"
  2. Chunked:      "{basename}_part000.npy", "{basename}_part001.npy", ...
                   + "{basename}_meta.json" (with a "chunks" manifest)

Chunked mode exists so large DEMs can be split into pieces small enough
to upload through GitHub's web UI (default target: 24.5MB/file) without
needing Git LFS. Each chunk is a contiguous row-slice of the full grid;
the manifest in the metadata records each chunk's file name and row
range, and this class stitches lookups across chunks transparently.

The public interface (sample / sample_many) is identical either way, so
calling code (e.g. coordinates_to_min_el.py) doesn't need to know or
care which layout is in use.
"""

import json
import numpy as np

# This DEM's CRS is "Simple Cylindrical" (Equirectangular), standard_parallel=0,
# central_meridian=0, on a sphere of radius R_MOON_M. That means the projected
# x/y (in meters) relate to lon/lat (in degrees) by a simple closed-form
# conversion - no pyproj needed:
#   x = R * radians(lon)
#   y = R * radians(lat)
R_MOON_M = 1737400.0  # matches SPHEROID["Moon", 1737400, 0] in the CRS


class LunarDEM:
    def __init__(self, basename):
        with open(f"{basename}_meta.json", "r") as f:
            self.meta = json.load(f)

        self.transform = self.meta["transform"]  # [a, b, c, d, e, f]
        self.a, self.b, self.c, self.d, self.e, self.f = self.transform
        self.bounds = self.meta["bounds"]
        self.shape = tuple(self.meta["shape"])

        self._basename = basename
        self.chunked = "chunks" in self.meta

        if self.chunked:
            # chunks: list of {"file": ..., "row_start": ..., "row_end": ...}
            # sorted by row_start (preprocessing writes them in order already,
            # but sort defensively in case a manifest is hand-edited)
            self.chunks = sorted(self.meta["chunks"], key=lambda c: c["row_start"])
            self._row_starts = np.array([c["row_start"] for c in self.chunks])
            self._row_ends = np.array([c["row_end"] for c in self.chunks])
            self._chunk_cache = {}  # lazy-loaded mmap arrays, keyed by chunk index

            total_rows = self.chunks[-1]["row_end"]
            if total_rows != self.shape[0]:
                raise ValueError(
                    f"Chunk manifest covers {total_rows} rows but metadata "
                    f"shape says {self.shape[0]}"
                )
        else:
            # Single-file layout (backward compatible)
            self.arr = np.load(f"{basename}.npy", mmap_mode="r")
            if self.arr.shape != self.shape:
                raise ValueError(
                    f"Array shape {self.arr.shape} doesn't match metadata {self.shape}"
                )

        # precompute the inverse-transform determinant once
        self._det = self.a * self.e - self.b * self.d
        if self._det == 0:
            raise ValueError("Affine transform is not invertible")

    # --- chunk management -------------------------------------------------

    def _chunk_index_for_row(self, row):
        """Return the index into self.chunks containing global row `row`,
        or None if out of range."""
        if row < 0 or row >= self.shape[0]:
            return None
        idx = int(np.searchsorted(self._row_starts, row, side="right") - 1)
        if idx < 0 or idx >= len(self.chunks):
            return None
        if row >= self._row_ends[idx]:
            return None
        return idx

    def _get_chunk_array(self, idx):
        """Lazily load (and cache) the mmap array for chunk `idx`."""
        if idx not in self._chunk_cache:
            fname = self.chunks[idx]["file"]
            self._chunk_cache[idx] = np.load(fname, mmap_mode="r")
        return self._chunk_cache[idx]

    # --- coordinate math ----------------------------------------------------

    def _lonlat_to_colrow(self, lon, lat):
        """
        Convert (lon, lat) in degrees to this DEM's projected x/y (meters),
        then invert the affine transform:
            x = a*col + b*row + c
            y = d*col + e*row + f
        Solve for (col, row) given (x, y).
        """
        a, b, c, d, e, f = self.a, self.b, self.c, self.d, self.e, self.f

        # Simple Cylindrical / Equirectangular projection (standard_parallel=0):
        x = R_MOON_M * np.radians(lon)
        y = R_MOON_M * np.radians(lat)

        col = (e * (x - c) - b * (y - f)) / self._det
        row = (a * (y - f) - d * (x - c)) / self._det

        return int(round(col)), int(round(row))

    # --- public sampling API ------------------------------------------------

    def sample(self, lon, lat):
        """
        Return the elevation value (meters) at (lon, lat), or None if
        out of bounds or nodata.
        """
        col, row = self._lonlat_to_colrow(lon, lat)

        if row < 0 or row >= self.shape[0] or col < 0 or col >= self.shape[1]:
            return None

        if self.chunked:
            idx = self._chunk_index_for_row(row)
            if idx is None:
                return None
            chunk_arr = self._get_chunk_array(idx)
            local_row = row - self.chunks[idx]["row_start"]
            val = chunk_arr[local_row, col]
        else:
            val = self.arr[row, col]

        if np.isnan(val):
            return None

        return float(val)

    def sample_many(self, coords):
        """
        Batched sampling. coords: list of (lon, lat) tuples.
        Returns a numpy array of elevations (np.nan where out of bounds/nodata).
        """
        a, b, c, d, e, f = self.a, self.b, self.c, self.d, self.e, self.f

        lons = np.array([p[0] for p in coords])
        lats = np.array([p[1] for p in coords])

        # Simple Cylindrical / Equirectangular projection (standard_parallel=0):
        xs = R_MOON_M * np.radians(lons)
        ys = R_MOON_M * np.radians(lats)

        cols = np.round((e * (xs - c) - b * (ys - f)) / self._det).astype(int)
        rows = np.round((a * (ys - f) - d * (xs - c)) / self._det).astype(int)

        valid = (
            (rows >= 0) & (rows < self.shape[0]) &
            (cols >= 0) & (cols < self.shape[1])
        )

        out = np.full(len(coords), np.nan, dtype=np.float64)

        if not self.chunked:
            out[valid] = self.arr[rows[valid], cols[valid]]
            return out

        # Chunked path: group the valid points by which chunk their row
        # falls into, so each chunk file is only opened/read once even if
        # many sample points land in it.
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            return out

        for chunk_idx, chunk in enumerate(self.chunks):
            row_start, row_end = chunk["row_start"], chunk["row_end"]
            in_chunk = valid_idx[
                (rows[valid_idx] >= row_start) & (rows[valid_idx] < row_end)
            ]
            if len(in_chunk) == 0:
                continue

            chunk_arr = self._get_chunk_array(chunk_idx)
            local_rows = rows[in_chunk] - row_start
            out[in_chunk] = chunk_arr[local_rows, cols[in_chunk]]

        return out

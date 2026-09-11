"""
lola_dem_reader.py

Runtime DEM reader for closed environments - depends only on numpy and
the standard library. Reads the .npy + _meta.json produced by
preprocess_lola_dem.py.
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

        # mmap so the whole array isn't loaded into RAM at once
        self.arr = np.load(f"{basename}.npy", mmap_mode="r")

        if self.arr.shape != self.shape:
            raise ValueError(
                f"Array shape {self.arr.shape} doesn't match metadata {self.shape}"
            )

        # precompute the inverse-transform determinant once
        self._det = self.a * self.e - self.b * self.d
        if self._det == 0:
            raise ValueError("Affine transform is not invertible")

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

    def sample(self, lon, lat):
        """
        Return the elevation value (meters) at (lon, lat), or None if
        out of bounds or nodata.
        """
        col, row = self._lonlat_to_colrow(lon, lat)

        if row < 0 or row >= self.shape[0] or col < 0 or col >= self.shape[1]:
            return None

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
        out[valid] = self.arr[rows[valid], cols[valid]]

        return out

"""
coordinates_to_min_el.py

Computes the minimum clear elevation angle (horizon mask) for 360 degrees
around a target coordinate on the Moon, using a preprocessed LOLA DEM.

Depends only on numpy + lola_dem_reader.py (itself numpy + stdlib only) -
no rasterio, no GDAL, no astropy required. Run preprocess_lola_dem.py
separately, on a machine with rasterio, to generate the .npy/.json files
this script loads.
"""

import numpy as np
from lola_dem_reader import LunarDEM

R_MOON_KM = 1737.4      # LOLA datum radius, kilometers
R_MOON_M = R_MOON_KM * 1000.0  # LOLA datum radius, meters


def translate_lunar_coordinate(lat1_deg, lon1_deg, distance_km, bearing_deg,
                                 r_moon_km=R_MOON_KM):
    """
    Given a starting point on the Moon, a distance (km), and a bearing
    (degrees, clockwise from north), returns the destination (lat, lon)
    in degrees. Standard spherical "direct geodesic" formula.
    """
    lat1 = np.radians(lat1_deg)
    lon1 = np.radians(lon1_deg)
    bearing = np.radians(bearing_deg)

    angular_dist = distance_km / r_moon_km  # central angle, radians

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular_dist)
        + np.cos(lat1) * np.sin(angular_dist) * np.cos(bearing)
    )

    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(angular_dist) * np.cos(lat1),
        np.cos(angular_dist) - np.sin(lat1) * np.sin(lat2)
    )

    return np.degrees(lat2), np.degrees(lon2)


def get_horizon_mask(target_lat, target_lon, dem_basename,
                      max_radius_km=50, num_points=360):
    """
    Finds the minimum clear elevation angle for 360 degrees around a
    coordinate, using a preprocessed LunarDEM (numpy-backed, no rasterio
    needed at runtime).
    """
    dem = LunarDEM(dem_basename)

    # --- Get target base elevation (H_target) ---
    h_target = dem.sample(target_lon, target_lat)
    if h_target is None:
        raise ValueError(
            f"Target ({target_lat}, {target_lon}) is out of DEM bounds or nodata"
        )
    print(f"h_target at ({target_lat}, {target_lon}): {h_target:.1f} m  "
          f"- sanity check this against a known reference elevation")

    horizon_angles = {}

    # --- Scan 360 degrees around the target point ---
    azimuths = np.arange(0, num_points) if num_points != 360 else np.arange(0, 360)

    for azimuth in azimuths:
        distances_km = np.arange(1, max_radius_km + 1)

        # Compute all radial points for this azimuth in one shot
        lats2, lons2 = translate_lunar_coordinate(
            target_lat, target_lon, distances_km, bearing_deg=azimuth
        )

        # Batched terrain sampling (much faster than per-point .sample calls)
        coords = list(zip(lons2, lats2))
        h_terrain = dem.sample_many(coords)

        # Math parameters
        delta_h = h_terrain - h_target
        d = distances_km * 1000.0  # convert to meters

        # Angle accounting for lunar curvature
        angle_rad = np.arctan2(delta_h - (d ** 2 / (2 * R_MOON_M)), d)
        angle_deg = np.degrees(angle_rad)

        # Worst-case (highest) obstruction in this direction, ignoring NaNs
        # from any out-of-bounds/nodata samples along the radial
        if np.all(np.isnan(angle_deg)):
            max_obstruction_angle = np.nan
        else:
            max_obstruction_angle = float(np.nanmax(angle_deg))

        horizon_angles[int(azimuth)] = max_obstruction_angle

    return horizon_angles


if __name__ == "__main__":
    # Example usage - adjust target coordinates and DEM basename as needed
    target_lat = -85.2
    target_lon = 31.6
    dem_basename = "ldem_118m"  # matches output_basename used in preprocessing

    mask = get_horizon_mask(target_lat, target_lon, dem_basename, max_radius_km=50)

    for az in sorted(mask.keys()):
        print(f"Azimuth {az:3d} deg: horizon elevation {mask[az]:.3f} deg")

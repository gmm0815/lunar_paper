"""
plot_3d_point_cloud.py

Renders a 3D point cloud of lunar coordinates from a results CSV (e.g.
results_100km.csv), color-coded by worst_angle_deg. Points are placed at
their real 3D position on the lunar sphere - computed from lat, lon, AND
elevation (h_target) - rather than just plotting lat/lon on two flat axes.

Since real lunar elevation changes (a few km) are tiny relative to the
Moon's ~1,737 km radius, an --exaggeration factor scales the elevation
deviation so terrain actually shows up as visible bumps rather than a
smooth, featureless sphere.

Rotate/zoom with the mouse: this already works in any matplotlib 3D
window opened with --show - left-click-drag rotates, scroll wheel (or
right-click-drag) zooms. No extra setup needed for that part.

Optional --basemap: pass a path to a real lunar surface image
(equirectangular projection, e.g. an LROC WAC mosaic) and this opens a
SECOND panel next to the point cloud, showing that image texture-mapped
onto a sphere. Rotating EITHER panel rotates both in sync, so you can
directly compare where your color-coded angle clusters land relative to
real craters/mountains/mare.

Uses matplotlib's built-in mplot3d - no new packages beyond what you
already have (pandas, matplotlib, numpy).

Usage:
    # Default: results_100km.csv, colored by worst_angle_deg
    python plot_3d_point_cloud.py

    # A different file / column
    python plot_3d_point_cloud.py results_50km.csv --color_column best_angle_deg

    # Tune terrain exaggeration (higher = more dramatic bumps)
    python plot_3d_point_cloud.py --exaggeration 50

    # Open an interactive rotatable window instead of just saving a PNG
    python plot_3d_point_cloud.py --show

    # Side-by-side with a real Moon texture, rotation synced between panels
    python plot_3d_point_cloud.py --show --basemap lroc_wac_mosaic.png

    # Set a specific static viewing angle for the saved PNG
    python plot_3d_point_cloud.py --elev 30 --azim 45
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed to register 3D projection

R_MOON_M = 1737400.0  # LOLA datum radius, meters - same constant used elsewhere in this project


def lonlat_elev_to_xyz(lat_deg, lon_deg, elevation_m, exaggeration=1.0):
    """
    Convert (lat, lon, elevation) to 3D Cartesian coordinates centered on
    the Moon. elevation_m is height above/below the R_MOON_M reference
    sphere (same convention as h_target throughout this project).
    exaggeration scales just the elevation deviation, not the base radius,
    so terrain is visible without distorting the overall sphere shape.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = R_MOON_M + exaggeration * elevation_m

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z


def build_data_surface(df, color_column, exaggeration=1.0):
    """
    Reconstruct the sampled points into a regular lat/lon grid (using the
    row/col indices already stored by full_grid_horizon_masks.py, since a
    strided sweep of a regular DEM grid IS itself a regular grid) and
    return 3D surface coordinates + a value grid to color it by.

    This avoids the "floating dots with gaps" look of a scatter plot -
    the result renders as one continuous surface, same as the real Moon
    texture panel.

    Row->lat and col->lon are recovered via a linear fit (exact for this
    equirectangular grid), so the surface's shape is still fully defined
    even if any individual grid cell's color value is missing (e.g. a
    nodata point that got skipped upstream).
    """
    rows_sorted = np.sort(df["row"].unique())
    cols_sorted = np.sort(df["col"].unique())
    n_r, n_c = len(rows_sorted), len(cols_sorted)

    a_lat, b_lat = np.polyfit(df["row"], df["lat"], 1)
    a_lon, b_lon = np.polyfit(df["col"], df["lon"], 1)

    lat_axis = a_lat * rows_sorted + b_lat
    lon_axis = a_lon * cols_sorted + b_lon
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)

    row_idx = np.searchsorted(rows_sorted, df["row"].values)
    col_idx = np.searchsorted(cols_sorted, df["col"].values)

    elev_grid = np.zeros((n_r, n_c))
    val_grid = np.full((n_r, n_c), np.nan)
    elev_grid[row_idx, col_idx] = df["h_target"].values
    val_grid[row_idx, col_idx] = df[color_column].values

    n_missing = np.isnan(val_grid).sum()
    if n_missing > 0:
        print(f"  Note: {n_missing} of {n_r * n_c} grid cells have no data "
              f"(rendered in gray)")

    x, y, z = lonlat_elev_to_xyz(lat_grid, lon_grid, elev_grid, exaggeration)
    return x, y, z, val_grid


def build_textured_sphere(basemap_path, n_lat=90, n_lon=180, extent=None):
    """
    Build a smooth sphere mesh (constant radius R_MOON_M) with facecolors
    sampled from a real lunar surface image, for side-by-side visual
    comparison against the color-coded point cloud.

    extent: (left, right, bottom, top) in degrees the image covers;
    defaults to the full Moon (-180, 180, -90, 90).
    """
    img = mpimg.imread(basemap_path)
    if img.ndim == 2:  # grayscale - broadcast to RGB
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:  # drop alpha if present
        img = img[..., :3]

    img_h, img_w = img.shape[0], img.shape[1]
    left, right, bottom, top = extent if extent else (-180, 180, -90, 90)

    lat_vals = np.linspace(90, -90, n_lat)   # top of image = north pole
    lon_vals = np.linspace(-180, 180, n_lon)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

    # Map each mesh cell's (lon, lat) to a pixel in the image
    col_idx = ((lon_grid - left) / (right - left) * (img_w - 1)).astype(int)
    row_idx = ((top - lat_grid) / (top - bottom) * (img_h - 1)).astype(int)
    col_idx = np.clip(col_idx, 0, img_w - 1)
    row_idx = np.clip(row_idx, 0, img_h - 1)

    facecolors = img[row_idx, col_idx] / 255.0 if img.dtype == np.uint8 else img[row_idx, col_idx]

    lat_r = np.radians(lat_grid)
    lon_r = np.radians(lon_grid)
    x = R_MOON_M * np.cos(lat_r) * np.cos(lon_r)
    y = R_MOON_M * np.cos(lat_r) * np.sin(lon_r)
    z = R_MOON_M * np.sin(lat_r)

    return x, y, z, facecolors


def sync_3d_views(fig, axes):
    """
    Link rotation across multiple 3D axes: dragging in any one of them
    applies the same azim/elev to all the others, so panels rotate
    together as if they were one shared camera.
    """
    state = {"syncing": False}

    def on_move(event):
        if state["syncing"] or event.inaxes not in axes:
            return
        source_ax = event.inaxes
        if source_ax.button_pressed not in source_ax._rotate_btn:
            return  # only sync while actively rotating (mouse button held)

        state["syncing"] = True
        for ax in axes:
            if ax is not source_ax:
                ax.view_init(elev=source_ax.elev, azim=source_ax.azim)
        fig.canvas.draw_idle()
        state["syncing"] = False

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def main():
    parser = argparse.ArgumentParser(
        description="3D point cloud of lunar coordinates, color-coded by a chosen angle column."
    )
    parser.add_argument("csv_path", nargs="?", default="results_100km.csv",
                         help="Path to the results CSV (default: results_100km.csv)")
    parser.add_argument("--color_column", default="worst_angle_deg",
                         help="Column to color points by (default: worst_angle_deg)")
    parser.add_argument("--exaggeration", type=float, default=20.0,
                         help="Elevation exaggeration factor for visibility (default: 20)")
    parser.add_argument("--point_size", type=float, default=8.0,
                         help="Scatter point size (default: 8.0)")
    parser.add_argument("--cmap", default="gray",
                         help="Matplotlib colormap (default: gray, to match the "
                              "greyscale look of the real lunar surface panel)")
    parser.add_argument("--output", default="point_cloud_3d.png",
                         help="Output PNG filename (default: point_cloud_3d.png)")
    parser.add_argument("--elev", type=float, default=20,
                         help="Viewing elevation angle in degrees for the saved PNG (default: 20)")
    parser.add_argument("--azim", type=float, default=45,
                         help="Viewing azimuth angle in degrees for the saved PNG (default: 45)")
    parser.add_argument("--show", action="store_true",
                         help="Open an interactive, rotatable matplotlib window "
                              "instead of (or in addition to) saving a static PNG")
    parser.add_argument("--basemap", default=None,
                         help="Path to a real lunar surface image (equirectangular) - "
                              "adds a second, rotation-synced panel showing it texture-"
                              "mapped onto a sphere, for direct visual comparison")
    parser.add_argument("--extent", type=float, nargs=4, default=None,
                         metavar=("LEFT", "RIGHT", "BOTTOM", "TOP"),
                         help="Basemap extent in degrees, if not the full Moon "
                              "(-180 180 -90 90)")
    parser.add_argument("--solid", action="store_true",
                         help="Render the data as a continuous colored surface "
                              "(reconstructed from row/col grid indices) instead "
                              "of scattered points with gaps between them")

    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    required = {"lat", "lon", "h_target", args.color_column}
    if args.solid:
        required |= {"row", "col"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {missing}. "
                          f"Columns present: {list(df.columns)}")

    df = df.dropna(subset=["lat", "lon", "h_target", args.color_column])

    if args.basemap:
        fig = plt.figure(figsize=(18, 9))
        ax_points = fig.add_subplot(121, projection="3d")
        ax_moon = fig.add_subplot(122, projection="3d")
    else:
        fig = plt.figure(figsize=(10, 9))
        ax_points = fig.add_subplot(111, projection="3d")
        ax_moon = None

    if args.solid:
        x, y, z, val_grid = build_data_surface(df, args.color_column, args.exaggeration)

        norm = plt.Normalize(vmin=np.nanmin(val_grid), vmax=np.nanmax(val_grid))
        cmap = plt.get_cmap(args.cmap)
        facecolors = cmap(norm(val_grid))
        facecolors[np.isnan(val_grid)] = (0.5, 0.5, 0.5, 1.0)  # gray for missing cells

        ax_points.plot_surface(
            x, y, z, facecolors=facecolors, rstride=1, cstride=1,
            shade=False, antialiased=False,
        )
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(val_grid)
        fig.colorbar(mappable, ax=ax_points, label=f"{args.color_column} (deg)", shrink=0.7)
    else:
        x, y, z = lonlat_elev_to_xyz(
            df["lat"].values, df["lon"].values, df["h_target"].values,
            exaggeration=args.exaggeration,
        )
        colors = df[args.color_column].values

        sc = ax_points.scatter(x, y, z, c=colors, cmap=args.cmap, s=args.point_size, depthshade=True)
        fig.colorbar(sc, ax=ax_points, label=f"{args.color_column} (deg)", shrink=0.7)

    ax_points.set_title(f"Point cloud - colored by {args.color_column}  (n={len(df)})")
    ax_points.set_xlabel("X (m)")
    ax_points.set_ylabel("Y (m)")
    ax_points.set_zlabel("Z (m)")

    # Roughly equal aspect so the sphere doesn't look stretched
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = x.mean(), y.mean(), z.mean()
    ax_points.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_points.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_points.set_zlim(mid_z - max_range, mid_z + max_range)
    ax_points.view_init(elev=args.elev, azim=args.azim)

    if args.basemap:
        mx, my, mz, facecolors = build_textured_sphere(args.basemap, extent=args.extent)
        ax_moon.plot_surface(
            mx, my, mz, facecolors=facecolors, rstride=1, cstride=1,
            shade=False, antialiased=False,
        )
        ax_moon.set_title("Real lunar surface (texture-mapped)")
        ax_moon.set_xlabel("X (m)")
        ax_moon.set_ylabel("Y (m)")
        ax_moon.set_zlabel("Z (m)")
        moon_range = R_MOON_M * 1.05
        ax_moon.set_xlim(-moon_range, moon_range)
        ax_moon.set_ylim(-moon_range, moon_range)
        ax_moon.set_zlim(-moon_range, moon_range)
        ax_moon.view_init(elev=args.elev, azim=args.azim)

        sync_3d_views(fig, [ax_points, ax_moon])
        print("Basemap panel added - rotating either panel will rotate both in sync.")

    plt.tight_layout()
    try:
        plt.savefig(args.output, dpi=150)
        print(f"Saved {args.output}  (n={len(df)} points, exaggeration={args.exaggeration}x)")
    except OSError as e:
        print(f"Warning: could not save {args.output} ({e}). "
              f"Continuing without saving - this is likely a Windows file lock "
              f"or a stale Pillow/matplotlib issue, not a problem with your data. "
              f"Try a different --output filename or `pip install --upgrade pillow matplotlib`.")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

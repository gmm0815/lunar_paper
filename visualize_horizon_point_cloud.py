"""
visualize_horizon_point_cloud.py

Reads the JSON output from full_grid_horizon_masks.py and renders a
color-coded point cloud - one point per (lat, lon) grid position,
colored by either the worst (most obstructed) or best (clearest)
horizon angle at that point.

The intent: plotted in the same lon/lat (equirectangular) space as a
real lunar surface image, valleys should show up as clusters of high
"worst angle" values (surrounded/obstructed) and mountain ranges should
show as low "worst angle" / distinctly different coloring from their
surroundings - a visual cross-check that the whole DEM -> horizon-mask
pipeline is behaving sensibly.

Requires matplotlib (not needed anywhere else in this project - this is
a standalone visualization/QC script, meant to run on your normal dev
machine, not the closed environment).

Usage:
    # Single field, plain point cloud:
    python visualize_horizon_point_cloud.py results_grid.json --field worst

    # Both worst and best side by side:
    python visualize_horizon_point_cloud.py results_grid.json --field both

    # Overlaid on a lunar basemap image (equirectangular projection,
    # full -180..180 / -90..90 extent assumed unless you pass --extent):
    python visualize_horizon_point_cloud.py results_grid.json --field worst \\
        --basemap lunar_albedo_map.png --alpha 0.6

    # Custom basemap extent (left, right, bottom, top) in degrees, if your
    # basemap image doesn't cover the full Moon:
    python visualize_horizon_point_cloud.py results_grid.json --field worst \\
        --basemap crop.png --extent -20 20 10 30
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_results(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    results = data["results"]
    lats = np.array([r["lat"] for r in results])
    lons = np.array([r["lon"] for r in results])
    worst = np.array([r["worst_angle_deg"] for r in results])
    best = np.array([r["best_angle_deg"] for r in results])

    print(f"Loaded {len(results)} points "
          f"(completed {data.get('completed', '?')} / {data.get('total_candidates', '?')})")
    print(f"  worst angle range: {np.nanmin(worst):.2f} to {np.nanmax(worst):.2f} deg")
    print(f"  best angle range:  {np.nanmin(best):.2f} to {np.nanmax(best):.2f} deg")

    return lats, lons, worst, best


def plot_field(ax, lats, lons, values, title, cmap, point_size,
                basemap_path=None, extent=None, alpha=1.0):
    if basemap_path:
        img = mpimg.imread(basemap_path)
        if extent is None:
            extent = [-180, 180, -90, 90]  # assume full-Moon equirectangular
        ax.imshow(img, extent=extent, cmap="gray", zorder=0)

    sc = ax.scatter(
        lons, lats, c=values, cmap=cmap, s=point_size,
        alpha=alpha, edgecolors="none", zorder=1,
    )
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(title)
    ax.set_aspect("equal")  # matches the equirectangular projection of the data itself
    plt.colorbar(sc, ax=ax, label="Angle (deg)")


def main():
    parser = argparse.ArgumentParser(
        description="Plot a color-coded horizon-angle point cloud from full_grid_horizon_masks.py output."
    )
    parser.add_argument("results_json", help="Path to the results JSON from full_grid_horizon_masks.py")
    parser.add_argument("--field", choices=["worst", "best", "both"], default="worst",
                         help="Which value to color by (default: worst)")
    parser.add_argument("--output", default="horizon_point_cloud.png",
                         help="Output image filename (default horizon_point_cloud.png)")
    parser.add_argument("--cmap", default="inferno",
                         help="Matplotlib colormap name (default: inferno)")
    parser.add_argument("--point_size", type=float, default=4.0,
                         help="Scatter point size (default 4.0 - shrink if points overlap too much)")
    parser.add_argument("--basemap", default=None,
                         help="Optional path to a lunar surface image to overlay points on")
    parser.add_argument("--extent", type=float, nargs=4, default=None,
                         metavar=("LEFT", "RIGHT", "BOTTOM", "TOP"),
                         help="Basemap extent in degrees, if not the full -180/180/-90/90 Moon")
    parser.add_argument("--alpha", type=float, default=1.0,
                         help="Point transparency (0-1), lower if overlaying a basemap")

    args = parser.parse_args()

    lats, lons, worst, best = load_results(args.results_json)

    if args.field == "both":
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        plot_field(axes[0], lats, lons, worst, "Worst (most obstructed) angle",
                   args.cmap, args.point_size, args.basemap, args.extent, args.alpha)
        plot_field(axes[1], lats, lons, best, "Best (clearest) angle",
                   args.cmap, args.point_size, args.basemap, args.extent, args.alpha)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        values = worst if args.field == "worst" else best
        title = "Worst (most obstructed) angle" if args.field == "worst" else "Best (clearest) angle"
        plot_field(ax, lats, lons, values, title,
                   args.cmap, args.point_size, args.basemap, args.extent, args.alpha)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()

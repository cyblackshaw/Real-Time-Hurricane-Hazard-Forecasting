"""Generate saved manuscript figures related to tropical-cyclone tracks."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from shapely.affinity import rotate

from config import (
    BDECK_DIR,
    COAST_DIR,
    EXCLUDED_YEARS,
    FIGURE_DIR,
    POSTPROCESS_DIR,
    STORM_IDS_BY_YEAR,
    YEARS,
)

PAIRED_TRACKS = POSTPROCESS_DIR / "track_position_paired.csv"
MAP_BOUNDS = (-99, -60, 24, 46)


def plot_position_categories_by_year(
    data,
    category_col,
    categories,
    colors,
    legend_labels,
    output,
    min_lead=6,
    max_lead=72,
):
    """Plot stacked BEST-position fractions by lead time and year."""
    df = data[data["lead_hour"].between(min_lead, max_lead)].copy()
    leads = np.sort(df["lead_hour"].dropna().unique())
    years = [y for y in YEARS if y not in EXCLUDED_YEARS]
    panels = years + ["All"]

    def summarize(sub):
        rows = []
        for lead in leads:
            values = sub.loc[np.isclose(sub["lead_hour"], lead), category_col].dropna()
            n = len(values)
            rows.append([100 * np.mean(values == c) if n else 0 for c in categories])
        return np.asarray(rows)

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(2.1 * len(panels), 5.5),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    ypos = np.arange(len(leads))

    for i, panel in enumerate(panels):
        ax = axes[i]
        summary = summarize(df if panel == "All" else df[df["year"] == panel])
        left = np.zeros(len(leads))
        for j, category in enumerate(categories):
            ax.barh(ypos, summary[:, j], left=left, height=0.8, color=colors[j], edgecolor="none")
            left += summary[:, j]

        ax.axvline(67, color="k", linestyle="--", alpha=0.5)
        ax.set_title(str(panel), fontsize=11)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.75, len(leads) - 0.25)
        ax.set_xticks([0, 50, 100])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}%"))
        ax.set_yticks(ypos)
        ax.tick_params(axis="x", labelrotation=90, length=0)
        ax.tick_params(axis="y", length=0, labelleft=(i == 0))
        if i == 0:
            ax.set_yticklabels([f"{int(x)}" for x in leads])
            ax.set_ylabel("Forecast Lead Time (h)")
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles = [Patch(facecolor=c, label=l) for c, l in zip(colors, legend_labels)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=len(handles), frameon=False)
    fig.supxlabel("Proportion of BEST Track Positions", y=0.02)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.15, top=0.88, wspace=0.12)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_track_position_figures():
    """Generate final paired CoU and DYN track-position figures."""
    paired = pd.read_csv(PAIRED_TRACKS)
    plot_position_categories_by_year(
        paired,
        "cone_position_class",
        ["cone1", "cone2", "outside"],
        ["#e06969", "#6999e0", "lightgrey"],
        ["Cone 1", "Cone 2", "Outside"],
        FIGURE_DIR / "ofcl_hind_tracks.png",
    )
    plot_position_categories_by_year(
        paired,
        "best_position",
        ["dense", "sparse", "outside"],
        ["mediumslateblue", "blueviolet", "lightgrey"],
        ["Dense", "Sparse", "Outside"],
        FIGURE_DIR / "dyn_hind_tracks.png",
    )


def rotate_gdf(gdf, angle=-30, origin=(-81.5, 27.5)):
    """Rotate geometries to match the manuscript map orientation."""
    out = gdf.copy()
    out["geometry"] = out.geometry.apply(lambda geom: rotate(geom, angle, origin=origin))
    return out


def parse_atcf_coordinates(track):
    lat = track["Lat"].str.replace(r"\D", "", regex=True).astype(float) / 10
    lon = -track["Lon"].str.replace(r"\D", "", regex=True).astype(float) / 10
    return lon, lat


def label_offset(x, y, bounds):
    """Place storm labels inward when their anchor lies near a map edge."""
    xmin, xmax, ymin, ymax = bounds
    dx, dy, ha, va = 5, 3, "left", "bottom"
    if x > xmax - 0.10 * (xmax - xmin):
        dx, ha = -5, "right"
    elif x < xmin + 0.10 * (xmax - xmin):
        dx, ha = 5, "left"
    if y > ymax - 0.10 * (ymax - ymin):
        dy, va = -4, "top"
    elif y < ymin + 0.10 * (ymax - ymin):
        dy, va = 4, "bottom"
    return dx, dy, ha, va


def resolve_annotation_overlap(fig, ax, annotations, step=4, max_shift=28):
    """Shift labels vertically to reduce overlap while keeping them on-map."""
    fig.canvas.draw()
    axes_bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    shifts = [0] + [s for d in range(step, max_shift + 1, step) for s in (d, -d)]
    placed = []

    for annotation in annotations:
        x0, y0 = annotation.get_position()
        for shift in shifts:
            annotation.set_position((x0, y0 + shift))
            fig.canvas.draw()
            bbox = annotation.get_window_extent(renderer=fig.canvas.get_renderer())
            overlaps = any(bbox.overlaps(existing) for existing in placed)
            inside = (
                bbox.x0 >= axes_bbox.x0
                and bbox.x1 <= axes_bbox.x1
                and bbox.y0 >= axes_bbox.y0
                and bbox.y1 <= axes_bbox.y1
            )
            if not overlaps and inside:
                break
        fig.canvas.draw()
        placed.append(annotation.get_window_extent(renderer=fig.canvas.get_renderer()))


def plot_observed_storm_tracks():
    """Plot observed storm tracks colored by intensity."""
    states = rotate_gdf(gpd.read_file(COAST_DIR / "cb_2018_us_state_20m/cb_2018_us_state_20m.shp"))
    ocean = rotate_gdf(gpd.read_file(COAST_DIR / "ne_10m_ocean/ne_10m_ocean.shp"))
    lakes = rotate_gdf(gpd.read_file(COAST_DIR / "great_lakes/GL250515_lam.shp").to_crs(4326))

    # Saffir-Simpson thresholds in m/s.
    cmap = mcolors.ListedColormap(["0.7", "#ffffcc", "#ffe775", "#ffc140", "#ff8f20", "#ff4040"])
    norm = mcolors.BoundaryNorm([0, 33, 43, 50, 59, 70, 100], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("lightgrey")
    states.plot(ax=ax, edgecolor="k", facecolor="lightgrey", linewidth=0.4, alpha=0.5)
    ocean.plot(ax=ax, facecolor="lightskyblue", alpha=0.4)
    lakes.plot(ax=ax, facecolor="lightskyblue", alpha=0.4)

    xmin, xmax, ymin, ymax = MAP_BOUNDS
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xticks=[], yticks=[], xlabel="", ylabel="")
    annotations = []

    for year in YEARS:
        for storm in STORM_IDS_BY_YEAR[year]:
            storm_id = f"{int(storm):02d}"
            track = pd.read_csv(BDECK_DIR / str(year) / f"al{storm_id}{year}.csv")
            track = track.loc[track["StormNum"] == int(storm)].copy()
            if track.empty:
                continue

            lon, lat = parse_atcf_coordinates(track)
            vmax_kts = track["Vmax [kts]"].astype(float)
            points = gpd.GeoDataFrame(track, geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
            points = rotate_gdf(points)
            points["vmax"] = vmax_kts.to_numpy() * 0.514444
            x, y = points.geometry.x.to_numpy(), points.geometry.y.to_numpy()

            ax.plot(x, y, color="k", linewidth=1.5, zorder=1)
            points.plot(
                ax=ax,
                column="vmax",
                cmap=cmap,
                norm=norm,
                markersize=np.where(vmax_kts.to_numpy() < 64, 7, 20),
                zorder=2,
            )

            valid_name = track["StormName"].notna() & track["StormName"].astype(str).str.strip().str.upper().ne("NAN")
            named = track.loc[(vmax_kts >= np.nanmax(vmax_kts) - 5) & valid_name]
            visible = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            if named.empty or not visible.any():
                continue

            idx = np.flatnonzero(visible)[-1]
            dx, dy, ha, va = label_offset(x[idx], y[idx], MAP_BOUNDS)
            annotations.append(
                ax.annotate(
                    f"{named['StormName'].iloc[0].strip()} ({year})",
                    xy=(x[idx], y[idx]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=7.5,
                    fontweight="bold",
                    ha=ha,
                    va=va,
                    color="k",
                    zorder=10,
                    clip_on=True,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.6),
                )
            )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        ticks=[16.5, 38, 46.5, 54.5, 64.5, 85],
        orientation="horizontal",
        shrink=0.6,
        aspect=30,
        pad=0.03,
    )
    cbar.ax.set_xticklabels(["TS or weaker", "Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"])
    resolve_annotation_overlap(fig, ax, annotations)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "storm_tracks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_track_schematic():
    """Create the conceptual CoU-versus-DYN track-classification schematic."""
    fig, axes = plt.subplots(2, 1, figsize=(4, 11), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.13, hspace=0.12)
    forecast_hours = np.array([0, 6, 12, 18])
    y_hr = np.array([1.0, 4.0, 6.5, 9.0])
    y_env = np.linspace(y_hr.min(), y_hr.max(), 300)
    obs_x, obs_y = np.array([0.0, 0.55, -0.2, 1.2]), np.array([1.0, 3.0, 5.2, 8.5])

    for ax in axes:
        ax.set(xlim=(-2.5, 2.6), ylim=(0.75, 9.25), xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
        for hr, yy in zip(forecast_hours, y_hr):
            ax.text(-2.38, yy, f"{hr} h", ha="left", va="center", fontsize=10, color="0.35")
        ax.plot(obs_x, obs_y, color="k", lw=3.5, marker="o", ms=6, zorder=10)

    # CoU panel.
    ofcl_x = np.array([0.0, 0.25, -0.3, 0.35])
    cone_width = np.array([0.05, 0.75, 1.35, 1.95])
    left, right = ofcl_x - cone_width, ofcl_x + cone_width
    left_i = np.interp(y_env, y_hr, left)
    right_i = np.interp(y_env, y_hr, right)
    center_i = np.interp(y_env, y_hr, ofcl_x)
    axes[0].fill_betweenx(y_env, -2.5, left_i, color="lightgrey", alpha=0.35)
    axes[0].fill_betweenx(y_env, left_i, center_i, color="#e06969", alpha=0.75)
    axes[0].fill_betweenx(y_env, center_i, right_i, color="#6999e0", alpha=0.75)
    axes[0].fill_betweenx(y_env, right_i, 2.6, color="lightgrey", alpha=0.35)
    axes[0].plot(ofcl_x, y_hr, "-o", color="0.35", lw=2, ms=5, linestyle=(0, (2, 3)), zorder=5)
    axes[0].plot(left, y_hr, "--", color="0.35", lw=1.2)
    axes[0].plot(right, y_hr, "--", color="0.35", lw=1.2)
    axes[0].set_title("(a) OFCL + Cone of Uncertainty", fontsize=12, fontweight="bold", pad=3)

    # Dynamical-ensemble panel.
    tracks = np.array([
        [0.0, -1.35, -1.65, -1.85], [0.0, -0.50, -1.05, -1.40],
        [0.0, -0.10, 0.15, -0.10], [0.0, 0.23, 0.38, 0.25],
        [0.0, 0.12, 0.48, 0.62], [0.0, 0.32, 0.63, 0.78],
        [0.0, 0.57, 0.39, 0.90], [0.0, 0.35, 1.25, 1.60],
    ])
    center = np.median(tracks, axis=0)
    spread_min, spread_max = tracks.min(axis=0), tracks.max(axis=0)
    central_min = center + 0.5 * (spread_min - center)
    central_max = center + 0.5 * (spread_max - center)
    spread_min_i = np.interp(y_env, y_hr, spread_min)
    spread_max_i = np.interp(y_env, y_hr, spread_max)
    central_min_i = np.interp(y_env, y_hr, central_min)
    central_max_i = np.interp(y_env, y_hr, central_max)

    axes[1].fill_betweenx(y_env, -2.5, spread_min_i, color="lightgrey", alpha=0.35)
    axes[1].fill_betweenx(y_env, spread_max_i, 2.6, color="lightgrey", alpha=0.35)
    axes[1].fill_betweenx(y_env, spread_min_i, central_min_i, color="blueviolet", alpha=0.16)
    axes[1].fill_betweenx(y_env, central_max_i, spread_max_i, color="blueviolet", alpha=0.16)
    axes[1].fill_betweenx(y_env, central_min_i, central_max_i, color="mediumslateblue", alpha=0.65)
    axes[1].plot(spread_min, y_hr, "--", color="0.35", lw=1.2)
    axes[1].plot(spread_max, y_hr, "--", color="0.35", lw=1.2)
    axes[1].plot(center, y_hr, color="indigo", lw=1.8, zorder=5)
    for track in tracks:
        axes[1].plot(track, y_hr, color="blueviolet", lw=1.5, linestyle=(0, (2, 3)), alpha=0.8, zorder=4)
    axes[1].set_title("(b) Dynamical Ensemble", fontsize=12, fontweight="bold", pad=3)

    handles = [
        Line2D([0], [0], color="k", lw=3, marker="o", label="Reference track"),
        Line2D([0], [0], color="0.35", lw=2, ls=":", label="OFCL"),
        Line2D([0], [0], color="blueviolet", lw=1.5, ls=":", label="DYN models"),
        Patch(facecolor="#e06969", alpha=0.75, label="CoU 1"),
        Patch(facecolor="#6999e0", alpha=0.75, label="CoU 2"),
        Patch(facecolor="mediumslateblue", alpha=0.65, label="DYN dense"),
        Patch(facecolor="blueviolet", alpha=0.16, label="DYN sparse"),
    ]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False, fontsize=8)
    fig.savefig(FIGURE_DIR / "hind_tracks_schematics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    make_track_position_figures()
    plot_observed_storm_tracks()
    plot_track_schematic()


if __name__ == "__main__":
    main()

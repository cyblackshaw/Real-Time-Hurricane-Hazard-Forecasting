"""Generate manuscript figures describing forecast revisions and their timing."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from config import ADECK_DIR, BDECK_DIR, FIGURE_DIR, HAZARD_LABELS, POSTPROCESS_DIR, STORM_IDS_BY_YEAR, YEARS

REVISION_FILE = POSTPROCESS_DIR / "all_hazard_forecast_change.csv"
HAZARDS = ("wind", "rain1", "surge", "power")
MARKERS = ("o", "^", "v", "d")
YEAR_TO_ROW = {
    **{year: 0 for year in range(2011, 2017)},
    **{year: 1 for year in range(2017, 2020)},
    2020: 2,
    **{year: 3 for year in range(2021, 2024)},
    2024: 4,
}


def storm_name(year, storm):
    """Read a storm name from B-deck data; fall back to year/storm number."""
    path = BDECK_DIR / str(year) / f"al{int(storm):02d}{year}.csv"
    try:
        track = pd.read_csv(path)
        vmax = track["Vmax [kts]"].astype(float)
        valid = track["StormName"].notna() & track["StormName"].astype(str).str.strip().str.upper().ne("NAN")
        named = track.loc[(vmax >= np.nanmax(vmax) - 5) & valid]
        if not named.empty:
            return str(named["StormName"].iloc[0]).strip()
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return f"Storm {int(storm)}"


def plot_forecast_change_summary(revisions):
    """Plot median/IQR OFCI magnitude, footprint, and replacement revisions."""
    metrics = [
        ("normalized_mae_change", r"$\Delta$nMAE", (-0.05, 0.7)),
        ("jaccard_change", r"$\mathbf{d_J}$", (-0.05, 0.6)),
        ("replacement_fraction", r"$\mathbf{R}$", (-0.05, 1.0)),
    ]
    data = revisions[(revisions["model"] == "OFCI") & (revisions["warning_hr"] <= 0)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

    for i, (metric, title, ylim) in enumerate(metrics):
        summary = (
            data.groupby(["hazard", "warning_hr"])[metric]
            .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
            .reset_index()
        )
        for j, hazard in enumerate(HAZARDS):
            sub = summary[summary["hazard"] == hazard].sort_values("warning_hr")
            line, = axes[i].plot(
                sub["warning_hr"], sub["median"], marker=MARKERS[j], linewidth=2.5,
                markersize=8, label=HAZARD_LABELS[hazard]
            )
            axes[i].fill_between(sub["warning_hr"], sub["q25"], sub["q75"], color=line.get_color(), alpha=0.15)
        axes[i].set(title=title, xlim=(-60, 0), ylim=ylim)
        axes[i].title.set_fontsize(20)
        axes[i].title.set_fontweight("bold")

    axes[1].legend(frameon=False, fontsize=15, loc="upper right")
    for i, (x, y) in enumerate([(-59, 0.615), (-59, 0.5275), (-59, 0.89)]):
        axes[i].text(x, y, f"({chr(97 + i)})", fontweight="bold", fontsize=20)
    fig.supxlabel("Forecast time relative to landfall (hr)", fontsize=20)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ofcl_hind_mae.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def peak_timing_table(revisions):
    """Return warning hour of maximum OFCI magnitude revision for each storm/hazard."""
    data = revisions[(revisions["model"] == "OFCI") & (revisions["warning_hr"] <= 0)].copy()
    idx = data.groupby(["year", "storm", "hazard"])["normalized_mae_change"].idxmax()
    peak = data.loc[idx, ["year", "storm", "hazard", "warning_hr", "normalized_mae_change"]]
    peak = peak.rename(columns={"warning_hr": "peak_warning_hr", "normalized_mae_change": "peak_change"})
    return peak.pivot(index=["year", "storm"], columns="hazard", values="peak_warning_hr").reindex(columns=HAZARDS)


def plot_peak_timing(peak):
    """Plot storm-by-storm timing of maximum revision for each hazard."""
    fig, ax = plt.subplots(figsize=(15, 4))
    for i, ((year, storm), row) in enumerate(peak.iterrows()):
        vals = row[list(HAZARDS)].dropna()
        if len(vals) > 1:
            ax.plot([i, i], [vals.min(), vals.max()], color="0.75", lw=1, zorder=1)
        for j, hazard in enumerate(HAZARDS):
            if pd.notna(row[hazard]):
                ax.scatter(i, row[hazard], s=100, marker=MARKERS[j], color=f"C{j}", alpha=0.7, zorder=2)

    ax.axhline(0, color="0.5", ls="--", lw=0.8)
    ax.axhline(-24, color="k", ls="--", lw=0.8)
    ax.axhline(-48, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(peak)))
    ax.set_xticklabels(
        [f"{storm_name(year, storm)}\n({int(year)})" for year, storm in peak.index],
        rotation=45,
        fontsize=8,
    )
    ax.set_ylabel("Hour of maximum forecast change\n(relative to landfall)")
    ax.text(-0.7, -27, "24 hours", fontsize=10)
    ax.text(-0.7, -52, "48 hours", fontsize=10)
    ax.set_xlim(-0.8, len(peak) - 0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ofcl_hind_peak_timing.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_peak_timing_distribution(peak):
    """Plot violin/box distributions of maximum-revision timing across storms."""
    data = [peak[h].dropna().to_numpy() for h in HAZARDS]
    positions = np.arange(len(HAZARDS))
    fig, ax = plt.subplots(figsize=(15, 4))

    violins = ax.violinplot(data, positions=positions, widths=0.65, showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(violins["bodies"]):
        body.set_facecolor(f"C{i}")
        body.set_edgecolor("none")
        body.set_alpha(0.5)

    boxes = ax.boxplot(
        data,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="0.3"),
        capprops=dict(color="0.5"),
    )
    for box in boxes["boxes"]:
        box.set_facecolor("white")
        box.set_edgecolor("0.3")

    rng = np.random.default_rng(42)
    for i, vals in enumerate(data):
        x = rng.normal(positions[i], 0.045, len(vals))
        ax.scatter(x, vals, s=38, marker=MARKERS[i], color=f"C{i}", alpha=0.85, edgecolors="none", zorder=3)

    ax.axhline(0, color="0.5", ls="--", lw=0.8)
    ax.axhline(-24, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(-48, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.text(3.32, -4, "Landfall", fontsize=12)
    ax.set(yticks=[-48, -24, 0], xticks=positions, xlim=(-0.6, len(HAZARDS) - 0.4))
    ax.set_xticklabels([HAZARD_LABELS[h] for h in HAZARDS], fontsize=20)
    ax.set_ylabel(r"Hour of max($\Delta$nMAE)", fontsize=20)
    ax.text(-0.575, -3.5, "(d)", fontweight="bold", fontsize=20)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ofcl_hind_peak_timing_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def calculate_storm_change(mat, year, storm):
    """Calculate OFCL track, intensity, and size revisions between advisories.

    This preserves the original notebook's matching behavior: rows are matched
    using the stored year/month/day/hour fields directly. The original notebook
    called this ``valid_time`` but did not add ``track_hours``.
    """
    fields = [
        "latstore", "lonstore", "vstore", "rstore", "yearstore", "monthstore",
        "daystore", "hourstore", "tc_num", "warning_hr", "track_hours",
    ]
    df = pd.DataFrame({key: np.asarray(mat[key]).ravel() for key in fields})
    df = df[(df["tc_num"] == storm) & (df["yearstore"] == year)].copy()
    df["match_time"] = pd.to_datetime(
        dict(year=df.yearstore, month=df.monthstore, day=df.daystore, hour=df.hourstore),
        errors="coerce",
    )
    df = df.dropna(subset=["match_time"]).drop_duplicates(["warning_hr", "match_time"])

    def haversine(lat1, lon1, lat2, lon2):
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dp, dl = p2 - p1, np.radians(lon2 - lon1)
        a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return 12742 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    def mean_abs_diff(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        valid = np.isfinite(a) & np.isfinite(b)
        return np.mean(np.abs(a[valid] - b[valid])) if valid.any() else np.nan

    rows = []
    warnings = np.arange(-72, 1, 6)
    for w1, w2 in zip(warnings[:-1], warnings[1:]):
        a = df[df["warning_hr"] == w1].set_index("match_time")
        b = df[df["warning_hr"] == w2].set_index("match_time")
        common = a.index.intersection(b.index)
        if common.empty:
            continue
        a, b = a.loc[common], b.loc[common]
        rows.append(
            {
                "warning_hr": w2,
                "track_change": np.nanmean(haversine(a.latstore, a.lonstore, b.latstore, b.lonstore)),
                "intensity_change": mean_abs_diff(a.vstore, b.vstore),
                "size_change": mean_abs_diff(a.rstore, b.rstore),
            }
        )
    return pd.DataFrame(rows)


def build_meteorological_revisions():
    """Calculate track/intensity/size revision time series for each storm."""
    changes = {}
    for year in YEARS:
        if not STORM_IDS_BY_YEAR[year]:
            continue
        mat = sio.loadmat(ADECK_DIR / f"{year}_land_ofcl_tracks_3day_FULL.mat")
        for storm in STORM_IDS_BY_YEAR[year]:
            changes[(year, storm)] = calculate_storm_change(mat, year, storm)
    return changes


def plot_peak_timing_heatmap(peak, storm_changes):
    """Compare timing of meteorological and hazard revision maxima by storm."""
    met_peak = {}
    for key, data in storm_changes.items():
        if data.empty:
            continue
        met_peak[key] = {
            col: data.loc[data[col].idxmax(), "warning_hr"] if data[col].notna().any() else np.nan
            for col in ("track_change", "intensity_change", "size_change")
        }
    met = pd.DataFrame.from_dict(met_peak, orient="index")
    met.index = pd.MultiIndex.from_tuples(met.index, names=peak.index.names)
    all_peak = peak.join(met, how="left")

    columns = ["track_change", "intensity_change", "size_change", *HAZARDS]
    labels = ["Track", "Intensity", "Size", *[HAZARD_LABELS[h] for h in HAZARDS]]
    matrix = all_peak[columns].T
    matrix.index = labels
    matrix.columns = [f"{storm_name(y, s)}\n({int(y)})" for y, s in all_peak.index]

    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="viridis",
        vmin=-72,
        vmax=0,
        mask=matrix.isna(),
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 8},
        cbar_kws={"label": "Hour of maximum forecast change\n(relative to landfall)"},
    )
    ax.axhline(3, color="white", lw=2.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set(xlabel="", ylabel="")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ofcl_peak_timing_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return all_peak


def revision_correlations(revisions, storm_changes):
    """Calculate within-storm Spearman correlation of meteorological/hazard revisions."""
    met_cols = {"track_change": "Track", "intensity_change": "Intensity", "size_change": "Size"}
    rows = []
    ofci = revisions[revisions["model"] == "OFCI"]

    for (year, storm), met in storm_changes.items():
        if met.empty:
            continue
        hazard = ofci[(ofci["year"] == year) & (ofci["storm"] == storm)]
        hwide = hazard.pivot_table(index="warning_hr", columns="hazard", values="normalized_mae_change", aggfunc="mean")
        merged = met.set_index("warning_hr").join(hwide, how="inner")

        for met_col, met_label in met_cols.items():
            for haz in HAZARDS:
                if haz not in merged:
                    continue
                pair = merged[[met_col, haz]].replace([np.inf, -np.inf], np.nan).dropna()
                rho = (
                    spearmanr(pair[met_col], pair[haz]).statistic
                    if len(pair) >= 4 and pair[met_col].nunique() > 1 and pair[haz].nunique() > 1
                    else np.nan
                )
                rows.append(
                    {
                        "year": year,
                        "storm": storm,
                        "meteorological": met_label,
                        "hazard": HAZARD_LABELS[haz],
                        "rho": rho,
                        "n": len(pair),
                        "t_met": pair[met_col].idxmax() if not pair.empty else np.nan,
                        "t_haz": pair[haz].idxmax() if not pair.empty else np.nan,
                    }
                )
    out = pd.DataFrame(rows)
    out["delta_t"] = out["t_haz"] - out["t_met"]
    return out


def plot_revision_correlations(comparison):
    """Plot Spearman-correlation distributions and coincident-revision timing."""
    met_order = ["Track", "Intensity", "Size"]
    haz_order = [HAZARD_LABELS[h] for h in HAZARDS]

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(
        data=comparison,
        x="hazard",
        y="rho",
        hue="meteorological",
        order=haz_order,
        hue_order=met_order,
        ax=ax,
        showfliers=False,
    )
    sns.stripplot(
        data=comparison,
        x="hazard",
        y="rho",
        hue="meteorological",
        order=haz_order,
        hue_order=met_order,
        dodge=True,
        ax=ax,
        alpha=0.35,
        size=3,
        legend=False,
    )
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set(ylim=(-1, 1), ylabel="Within-storm Spearman correlation", xlabel="")
    ax.legend(title="Meteorological revision", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "spearman_storm_haz.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    timing = (
        comparison.assign(same_advisory=lambda d: d["delta_t"] == 0)
        .pivot_table(index="meteorological", columns="hazard", values="same_advisory", aggfunc="mean")
        .reindex(index=met_order, columns=haz_order)
        * 100
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        timing,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Storms with coincident\nmaximum revisions (%)"},
    )
    ax.set(xlabel="", ylabel=" ")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "timing_storm_haz.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_storm_revision_panels(revisions):
    """Plot normalized OFCI hazard revision for each storm (original panel layout)."""
    nrows, ncols = 5, len(STORM_IDS_BY_YEAR)
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 10))
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.09, top=0.95, wspace=0.18, hspace=0.38)
    next_col = {row: 0 for row in range(nrows)}
    data = revisions[(revisions["model"] == "OFCI") & (revisions["warning_hr"] <= 0)]

    for year in YEARS:
        if not STORM_IDS_BY_YEAR[year]:
            continue
        row = YEAR_TO_ROW[year]
        for storm in STORM_IDS_BY_YEAR[year]:
            col = next_col[row]
            ax = axes[row, col]
            storm_data = data[(data["year"] == year) & (data["storm"] == storm)]
            for i, hazard in enumerate(HAZARDS):
                sub = storm_data[storm_data["hazard"] == hazard].sort_values("warning_hr")
                ax.plot(sub["warning_hr"], sub["normalized_mae_change"], marker=MARKERS[i], linewidth=2, zorder=2)
            if row == 0 and col == 0:
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            else:
                ax.tick_params(labelleft=False)
            ax.set(xlabel="", ylabel="", ylim=(0, 1))
            ax.set_title(f"{storm_name(year, storm)} ({year})")
            next_col[row] += 1

    # Dedicated legend panel.
    legend_ax = axes[0, 5]
    handles = [
        Line2D([0], [0], color=f"C{i}", marker=MARKERS[i], linewidth=3, label=HAZARD_LABELS[h])
        for i, h in enumerate(HAZARDS)
    ]
    legend_ax.legend(handles=handles, frameon=False, loc="center", fontsize=10)
    legend_ax.axis("off")

    for ax in axes.flat:
        if ax is not legend_ax and not ax.has_data():
            ax.remove()

    active = [ax for ax in axes.flat if ax in fig.axes and ax.has_data() and ax is not legend_ax]
    xmin = min(ax.get_position().x0 for ax in active)
    xmax = max(ax.get_position().x1 for ax in active)
    fig.supxlabel("Forecast lead time (hours)", fontsize=13, x=(xmin + xmax) / 2, y=0.02, fontweight="bold")
    fig.supylabel("Normalized forecast revision", fontsize=13, x=0.025, fontweight="bold")
    fig.savefig(FIGURE_DIR / "mean_haz_change_forecast.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    revisions = pd.read_csv(REVISION_FILE)
    plot_forecast_change_summary(revisions)
    peak = peak_timing_table(revisions)
    plot_peak_timing(peak)
    plot_peak_timing_distribution(peak)
    plot_storm_revision_panels(revisions)

    storm_changes = build_meteorological_revisions()
    plot_peak_timing_heatmap(peak, storm_changes)
    comparison = revision_correlations(revisions, storm_changes)
    plot_revision_correlations(comparison)


if __name__ == "__main__":
    main()

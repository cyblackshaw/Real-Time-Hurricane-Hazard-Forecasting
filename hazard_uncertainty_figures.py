"""Generate manuscript figures for hazard uncertainty, containment, and verification.

All non-track figures in the original ``summary_statistics``, ``uncertainty_comp``
and ``false_positives`` notebooks can be generated from one processed table:
``postprocess/state_all_UP_2011_2024.csv``.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
from shapely.affinity import rotate

from config import COAST_DIR, FIGURE_DIR, HAZARD_LABELS, POSTPROCESS_DIR

STATE_FILE = POSTPROCESS_DIR / "state_all_UP_2011_2024.csv"
HAZARDS = ("wind", "rain1", "surge", "power")
DYN_RANGE = "full"  # use "67" for the central 67% DYN interval
STATUS_ORDER = [-1, 0, 1]
STATUS_LABELS = ["+", "✓", "―"]


def normalize_geoid(series):
    """Normalize county identifiers read from CSV/shapefile inputs."""
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)


def summarize_across_storms(df, metrics, within_storm=None):
    """Summarize storm/advisory metrics with equal weighting across storms.

    ``within_storm`` optionally maps metric names to county-level aggregations
    (for example median width or mean containment) before the across-storm IQR
    is calculated.
    """
    data = df.copy()
    storm_keys = ["year", "tc_num", "hazard", "warning_hr"]
    if within_storm:
        data = data.groupby(storm_keys, as_index=False).agg(within_storm)

    rows = []
    for (hazard, warning), group in data.groupby(["hazard", "warning_hr"]):
        row = {"hazard": hazard, "warning_hr": warning}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_q25"] = values.quantile(0.25)
            row[f"{metric}_q75"] = values.quantile(0.75)
            row[f"{metric}_n"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)


def spatial_metrics(state):
    """Derive storm/advisory extent and Jaccard metrics directly from state data."""
    rows = []
    keys = ["year", "tc_num", "hazard", "warning_hr"]
    for key, group in state.groupby(keys):
        nhc = set(group.loc[group["nhc_possible"], "GEOID"])
        dyn = set(group.loc[group["dyn_possible"], "GEOID"])
        dyn67 = set(group.loc[group["dyn_p835"] > 0, "GEOID"])

        def extent_difference(a, b):
            denom = len(a) + len(b)
            return (len(a) - len(b)) / denom if denom else np.nan

        def jaccard(a, b):
            union = a | b
            return len(a & b) / len(union) if union else np.nan

        rows.append(
            dict(
                zip(keys, key),
                extent_difference_norm=extent_difference(dyn, nhc),
                extent67_difference_norm=extent_difference(dyn67, nhc),
                footprint_jaccard=jaccard(dyn, nhc),
                footprint67_jaccard=jaccard(dyn67, nhc),
            )
        )
    return pd.DataFrame(rows)


def plot_metric_by_advisory(
    summary,
    metrics,
    output,
    labels=None,
    hazards=HAZARDS,
    ylabel=None,
    ylim=None,
    zero_line=False,
    show_titles=True,
    shared_xlabel=False,
    show_legend=False,
):
    """Plot one or more uncertainty metrics across forecast advisory time."""
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    labels = labels or {m: m for m in metrics}
    colors = {
        "nhc_contains": "C0",
        "dyn_envelope_contains": "C1",
        "dyn_67_contains": "C2",
        "width_difference_norm": "C1",
        "width67_difference_norm": "C2",
        "extent_difference_norm": "C1",
        "extent67_difference_norm": "C2",
        "footprint_jaccard": "C1",
        "footprint67_jaccard": "C2",
    }

    fig, axes = plt.subplots(1, len(hazards), figsize=(4 * len(hazards), 3.8), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for i, hazard in enumerate(hazards):
        ax = axes[i]
        sub = summary[summary["hazard"] == hazard].sort_values("warning_hr")
        for j, metric in enumerate(metrics):
            color = colors.get(metric, f"C{j}")
            ax.plot(
                sub["warning_hr"],
                sub[f"{metric}_median"],
                "-o",
                color=color,
                linewidth=2.5,
                markersize=4,
                label=labels.get(metric, metric),
            )
            ax.fill_between(
                sub["warning_hr"],
                sub[f"{metric}_q25"],
                sub[f"{metric}_q75"],
                color=color,
                alpha=0.15,
            )
        if zero_line:
            ax.axhline(0, color="0.4", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="0.6", linestyle=":", linewidth=0.8)
        ax.set_xticks(np.arange(-66, 1, 12))
        if show_titles:
            ax.set_title(HAZARD_LABELS[hazard], fontweight="bold", fontsize=20)
        if ylim:
            ax.set_ylim(*ylim)
        if i == 0 and ylabel:
            ax.set_ylabel(ylabel, fontsize=20)
        ax.spines[["top", "right"]].set_visible(False)

    if shared_xlabel:
        fig.supxlabel("Forecast time relative to landfall (hr)", fontsize=20)
    if show_legend:
        axes[-1].legend(frameon=False, fontsize=15, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_summary_figures(state):
    """Generate width, footprint, Jaccard, and containment summary figures."""
    window = state[state["warning_hr"].between(-70, 6, inclusive="neither")].copy()
    width_summary = summarize_across_storms(
        window,
        ["width_difference_norm", "width67_difference_norm"],
        within_storm={"width_difference_norm": "median", "width67_difference_norm": "median"},
    )
    spatial = spatial_metrics(window)
    extent_summary = summarize_across_storms(spatial, ["extent_difference_norm", "extent67_difference_norm"])
    jaccard_summary = summarize_across_storms(spatial, ["footprint_jaccard", "footprint67_jaccard"])

    plot_metric_by_advisory(
        width_summary,
        ["width_difference_norm", "width67_difference_norm"],
        FIGURE_DIR / "haz_width_diff_w67.png",
        labels={"width_difference_norm": " ", "width67_difference_norm": " "},
        ylabel=r"$n\Delta W$",
        ylim=(-1, 1),
        zero_line=True,
    )
    plot_metric_by_advisory(
        extent_summary,
        ["extent_difference_norm", "extent67_difference_norm"],
        FIGURE_DIR / "haz_extent_diff_w67.png",
        labels={"extent_difference_norm": " ", "extent67_difference_norm": " "},
        ylabel=r"$n\Delta F$",
        ylim=(-1, 1),
        zero_line=True,
        show_titles=False,
    )
    plot_metric_by_advisory(
        jaccard_summary,
        ["footprint_jaccard", "footprint67_jaccard"],
        FIGURE_DIR / "haz_jacc_sim_w67.png",
        labels={"footprint_jaccard": r"DYN$_F$", "footprint67_jaccard": r"DYN$_{67}$"},
        ylabel="J\n",
        ylim=(0, 1),
        show_titles=False,
        shared_xlabel=True,
        show_legend=True,
    )

    contain = window[window["obs"] > 0].copy()
    contain["nhc_contains"] = contain["obs"].between(contain["nhc_min"], contain["nhc_max"])
    contain["dyn_envelope_contains"] = contain["obs"].between(contain["dyn_min"], contain["dyn_max"])
    contain["dyn_67_contains"] = contain["obs"].between(contain["dyn_p165"], contain["dyn_p835"])
    containment_summary = summarize_across_storms(
        contain,
        ["nhc_contains", "dyn_envelope_contains", "dyn_67_contains"],
        within_storm={
            "nhc_contains": "mean",
            "dyn_envelope_contains": "mean",
            "dyn_67_contains": "mean",
        },
    )
    plot_metric_by_advisory(
        containment_summary,
        ["nhc_contains", "dyn_envelope_contains", "dyn_67_contains"],
        FIGURE_DIR / "fraction_contain_haz.png",
        labels={"nhc_contains": "CoU", "dyn_envelope_contains": r"DYN$_F$", "dyn_67_contains": r"DYN$_{67}$"},
        ylabel="Containment\nFraction",
        ylim=(0, 1.05),
        shared_xlabel=True,
        show_legend=True,
    )


def reference_position(obs, low, center, high):
    """Normalize reference position relative to asymmetric forecast bounds."""
    obs, low, center, high = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in (obs, low, center, high)])
    denom = np.where(obs < center, center - low, high - center)
    valid = np.isfinite(obs) & np.isfinite(low) & np.isfinite(center) & np.isfinite(high) & (low <= center) & (center <= high) & (denom > 0)
    return np.divide(obs - center, denom, out=np.full(obs.shape, np.nan), where=valid)


def prepare_reference_positions(state):
    """Classify BEST position within CoU and DYN uncertainty ranges."""
    data = state[(state["obs"] > 0) & state["warning_hr"].between(-70, 6, inclusive="neither")].copy()
    data["nhc_position"] = reference_position(data.obs, data.nhc_min, data.ofcl, data.nhc_max)
    categories = ["Below", "Lower", "Central", "Upper", "Above"]

    p = -data["nhc_position"]
    data["nhc_class"] = np.where(
        p.notna(),
        np.select([p < -1, p < -0.5, p <= 0.5, p <= 1, p > 1], categories, default="Missing"),
        "Missing",
    )

    dyn_p = 100 - data["best_dyn_percentile"]
    valid = data.best_dyn_percentile.notna() & data.dyn_min.notna() & data.dyn_max.notna() & (data.dyn_min <= data.dyn_max)
    dyn_class = np.select(
        [data.obs > data.dyn_max, data.obs < data.dyn_min, dyn_p < 33.333, dyn_p <= 66.667],
        ["Below", "Above", "Lower", "Central"],
        default="Upper",
    )
    data["dyn_class"] = np.where(valid, dyn_class, "Missing")
    return data, categories


def storm_position_fractions(data, column, categories):
    keys = ["year", "tc_num", "hazard", "warning_hr"]
    valid = data.loc[data[column] != "Missing", keys + [column]]
    counts = pd.crosstab([valid[k] for k in keys], valid[column]).reindex(columns=categories, fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0).reset_index()


def plot_reference_position_figures(state):
    """Generate the two BEST-position-within-uncertainty figures."""
    position, categories = prepare_reference_positions(state)
    colors = ["#b2182b", "#ef8a62", "#d9d9d9", "#67a9cf", "#2166ac"]
    summaries = {
        "CoU": storm_position_fractions(position, "nhc_class", categories),
        "Full DYN": storm_position_fractions(position, "dyn_class", categories),
    }

    fig, axes = plt.subplots(2, len(HAZARDS), figsize=(15, 6), sharex=True, sharey=True)
    for i, (method, summary) in enumerate(summaries.items()):
        for j, hazard in enumerate(HAZARDS):
            ax = axes[i, j]
            sub = summary[summary["hazard"] == hazard].groupby("warning_hr")[categories].mean().sort_index()
            if not sub.empty:
                ax.stackplot(sub.index, *[sub[c].values for c in categories], colors=colors, labels=categories)
            if i == 0:
                ax.set_title(HAZARD_LABELS[hazard])
            if j == 0:
                ax.set_ylabel(f"{method}\nFraction")
            if i == 1:
                ax.set_xlabel("Advisory time relative to landfall (hr)")
            ax.set_ylim(0, 1)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURE_DIR / "reference_position_percentile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(HAZARDS), figsize=(16, 3.5), sharey=True)
    bins = np.arange(0, 105, 5)
    for ax, hazard in zip(axes, HAZARDS):
        d = position[position["hazard"] == hazard].copy()
        d = d[np.isfinite(d.obs) & np.isfinite(d.dyn_min) & np.isfinite(d.dyn_max) & (d.dyn_min <= d.dyn_max)]
        below = d.obs > d.dyn_max
        above = d.obs < d.dyn_min
        inside = ~(below | above) & d.best_dyn_percentile.notna()
        counts, _ = np.histogram(100 - d.loc[inside, "best_dyn_percentile"], bins=bins)
        total = len(d)
        fractions = counts / total if total else counts.astype(float)
        ax.bar(bins[:-1] + 2.5, fractions, width=4.5, color="C1", alpha=0.8)
        ax.bar(-7.5, below.mean() if total else 0, width=5, color="#2166ac")
        ax.bar(107.5, above.mean() if total else 0, width=5, color="#b2182b")
        ax.axvline(50, color="k", ls="--", lw=1)
        ax.axvline(16.5, color="0.5", ls=":", lw=1)
        ax.axvline(83.5, color="0.5", ls=":", lw=1)
        ax.set_xlim(-13, 113)
        ax.set_xticks([-7.5, 0, 25, 50, 75, 100, 107.5])
        ax.set_xticklabels(["DYN\nBelow", "0", "25", "50", "75", "100", "DYN\nAbove"])
        ax.set_title(HAZARD_LABELS[hazard])
        ax.set_xlabel("DYN predictions above reference (%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fraction of all valid cases")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "dyn_above_reference_percentile.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def safe_div(num, den):
    num, den = np.asarray(num), np.asarray(den)
    return np.divide(num, den, out=np.full(np.broadcast(num, den).shape, np.nan, dtype=float), where=den != 0)


def verification_tables(state):
    """Calculate county-level threshold verification and storm/advisory metrics."""
    thresholds = {"wind": 17, "rain1": 7.62, "surge": 0.55, "power": 1}
    bounds = {"CoU": "nhc_max", "DYN_67": "dyn_p835", "DYN_F": "dyn_max"}
    data = state[state["hazard"].isin(thresholds) & state["warning_hr"].between(-70, 6, inclusive="neither")].copy()
    data["threshold"] = data["hazard"].map(thresholds)
    data = data.dropna(subset=["obs", "threshold", *bounds.values()])
    data["reference_positive"] = data["obs"] >= data["threshold"]

    county_results = []
    id_cols = ["year", "tc_num", "hazard", "warning_hr", "GEOID"]
    for method, column in bounds.items():
        group = data[id_cols + ["reference_positive", "threshold", column]].copy()
        group["method"] = method
        group["forecast_positive"] = group[column] >= group["threshold"]
        p, r = group.forecast_positive, group.reference_positive
        group["TP"] = (p & r).astype(int)
        group["FP"] = (p & ~r).astype(int)
        group["FN"] = (~p & r).astype(int)
        group["TN"] = (~p & ~r).astype(int)
        county_results.append(group)

    county = pd.concat(county_results, ignore_index=True)
    counts = county.groupby(["year", "tc_num", "hazard", "warning_hr", "method"], as_index=False)[["TP", "FP", "FN", "TN"]].sum()
    counts["POD"] = safe_div(counts.TP, counts.TP + counts.FN)
    counts["FAR"] = safe_div(counts.FP, counts.TP + counts.FP)
    counts["CSI"] = safe_div(counts.TP, counts.TP + counts.FP + counts.FN)
    counts["FPR"] = safe_div(counts.FP, counts.FP + counts.TN)

    # Additional DYN_F positives relative to DYN_67.
    wide = data[[*id_cols, "reference_positive", "threshold", "dyn_p835", "dyn_max"]].copy()
    new_positive = (wide.dyn_max >= wide.threshold) & ~(wide.dyn_p835 >= wide.threshold)
    wide["additional_TP"] = (new_positive & wide.reference_positive).astype(int)
    wide["additional_FP"] = (new_positive & ~wide.reference_positive).astype(int)
    incremental = wide.groupby(["year", "tc_num", "hazard", "warning_hr"], as_index=False)[["additional_TP", "additional_FP"]].sum()
    return counts, incremental


def plot_verification_figures(state):
    """Generate POD/FAR/CSI advisory figures and added-TP/FP figure."""
    counts, incremental = verification_tables(state)
    methods = ["CoU", "DYN_F", "DYN_67"]
    names = {"CoU": "CoU", "DYN_F": r"DYN$_F$", "DYN_67": r"DYN$_{67}$"}
    colors = {"CoU": "tab:blue", "DYN_67": "tab:green", "DYN_F": "tab:orange"}

    advisory = (
        counts.groupby(["hazard", "method", "warning_hr"])[["POD", "FAR", "CSI", "FPR"]]
        .agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    )
    advisory.columns = [
        "_".join([a, {"<lambda_0>": "q25", "<lambda_1>": "q75"}.get(b, b)])
        for a, b in advisory.columns
    ]
    advisory = advisory.reset_index()

    for metric in ("POD", "FAR", "CSI"):
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
        for ax, hazard in zip(axes, HAZARDS):
            for method in methods:
                sub = advisory[(advisory.hazard == hazard) & (advisory.method == method)].sort_values("warning_hr")
                x = sub.warning_hr.to_numpy()
                ax.plot(x, sub[f"{metric}_median"], "-o", color=colors[method], lw=2, label=names[method])
                ax.fill_between(x, sub[f"{metric}_q25"], sub[f"{metric}_q75"], color=colors[method], alpha=0.15)
            if metric == "POD":
                ax.set_title(HAZARD_LABELS[hazard], fontsize=20, fontweight="bold")
            ax.set(ylim=(0, 1.05), xlim=(-67, 1), xticks=np.arange(-66, 1, 12))
        axes[0].set_ylabel(metric, fontsize=20)
        if metric == "CSI":
            axes[-1].legend(frameon=False, fontsize=15, loc="lower right")
            fig.supxlabel("Forecast time relative to landfall (hr)", fontsize=20)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{metric}_by_advisory.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for i, (ax, hazard) in enumerate(zip(axes, HAZARDS)):
        sums = incremental[incremental.hazard == hazard][["additional_TP", "additional_FP"]].sum()
        ax.bar(["+TP", "+FP"], sums.values, color=["tab:green", "tab:red"])
        ax.tick_params(axis="x", labelsize=15)
        ax.set_title(HAZARD_LABELS[hazard], fontsize=20, fontweight="bold")
        if i == 0:
            ax.set_ylabel("#Cases", fontsize=20)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "additional_TP_FP.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def classify_interval_relative_to_obs(obs, low, high):
    """Classify interval as overforecast (-1), containing (0), or underforecast (+1)."""
    obs, low, high = map(lambda x: np.asarray(x, dtype=float), (obs, low, high))
    status = np.full(len(obs), np.nan)
    valid = np.isfinite(obs) & np.isfinite(low) & np.isfinite(high) & (obs > 0)
    status[valid & (obs >= low) & (obs <= high)] = 0
    status[valid & (obs < low)] = -1
    status[valid & (obs > high)] = 1
    return status


def prepare_containment(state, dyn_range=DYN_RANGE):
    """Create CoU/DYN directional containment classes from the state table."""
    data = state[state["warning_hr"] < 6].copy()
    data["haz"] = data["hazard"]
    data["storm"] = data["year"].astype(str) + "_" + data["tc_num"].astype(str)
    data["GEOID"] = normalize_geoid(data["GEOID"])
    low, high = ("dyn_min", "dyn_max") if dyn_range == "full" else ("dyn_p165", "dyn_p835")
    data["dyn_status"] = classify_interval_relative_to_obs(data.obs, data[low], data[high])
    data["nhc_status"] = classify_interval_relative_to_obs(data.obs, data.nhc_min, data.nhc_max)
    valid = data.dyn_status.notna() & data.nhc_status.notna()
    data["containment"] = np.nan
    data.loc[valid, "containment"] = (
        (data.loc[valid, "nhc_status"] == 0).astype(int)
        + 2 * (data.loc[valid, "dyn_status"] == 0).astype(int)
    )
    return data


def joint_status_table(data, hazard, warning_hr=None):
    """Return normalized DYN x CoU status table in a fixed status order."""
    sub = data[data["haz"] == hazard]
    if warning_hr is not None:
        sub = sub[sub["warning_hr"] == warning_hr]
    table = pd.crosstab(sub.dyn_status, sub.nhc_status).reindex(index=STATUS_ORDER, columns=STATUS_ORDER, fill_value=0)
    total = table.to_numpy().sum()
    return table / total if total else table.astype(float)


def plot_miscontainment_heatmaps(data, dyn_range=DYN_RANGE):
    """Generate directional miscontainment heatmaps from one shared helper."""
    range_label = "F" if dyn_range == "full" else dyn_range
    hazards = ["wind", "rain1", "surge", "power"]
    titles = [HAZARD_LABELS[h] for h in hazards]
    tables = [joint_status_table(data, h, -48) for h in hazards]

    # 2x2 layout.
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.40])
    for k, ax in enumerate(axes.flat):
        sns.heatmap(
            tables[k] * 100,
            annot=True,
            cmap="YlGnBu",
            vmin=0,
            vmax=50,
            cbar=(k == 3),
            cbar_ax=cbar_ax if k == 3 else None,
            annot_kws={"size": 16},
            xticklabels=STATUS_LABELS,
            yticklabels=STATUS_LABELS,
            ax=ax,
        )
        ax.tick_params(axis="both", labelsize=16, rotation=0)
        ax.set_title(titles[k], fontweight="bold", fontsize=20)
        ax.set(xlabel="", ylabel="")
    axes[1, 0].set_ylabel(rf"$\mathbf{{DYN_{{{range_label}}}}}$", fontweight="bold", fontsize=20)
    axes[1, 0].set_xlabel(r"$\mathbf{CoU}$", fontweight="bold", fontsize=20)
    cbar_ax.set_ylabel("% Cases", rotation=270, labelpad=18)
    fig.savefig(FIGURE_DIR / f"dir_miscontainment_haz_{dyn_range}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 1x4 layout of the same -48 h results.
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    cbar_ax = fig.add_axes([0.91, 0.175, 0.01, 0.60])
    for k, ax in enumerate(axes):
        sns.heatmap(
            tables[k] * 100,
            annot=True,
            cmap="YlGnBu",
            vmin=0,
            vmax=50,
            cbar=(k == 3),
            cbar_ax=cbar_ax if k == 3 else None,
            annot_kws={"size": 16},
            xticklabels=STATUS_LABELS,
            yticklabels=STATUS_LABELS,
            ax=ax,
        )
        ax.tick_params(axis="both", labelsize=16, rotation=0)
        ax.set_title(titles[k], fontweight="bold", fontsize=20)
        ax.set(xlabel="", ylabel="")
    axes[0].set_ylabel(rf"$\mathbf{{DYN_{{{range_label}}}}}$", fontweight="bold", fontsize=20)
    axes[0].set_xlabel(r"$\mathbf{CoU}$", fontweight="bold", fontsize=20)
    cbar_ax.set_title("%", fontsize=20, pad=12)
    fig.savefig(FIGURE_DIR / f"dir_miscontainment_haz_{dyn_range}_1d.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Hazard x advisory matrix.
    warning_hours = [-66, -48, -36, -24, -12, -6, 0]
    hazards_adv = ["wind", "rain1", "rainT", "surge", "power"]
    labels_adv = ["Wind", "Hourly\nRain", "Accumulated\nRain", "Storm Tide", "Power\nOutage"]
    fig, axes = plt.subplots(len(hazards_adv), len(warning_hours), figsize=(19, 11), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.1, right=0.89, bottom=0.12, top=0.88, wspace=0.12, hspace=0.25)
    cbar_ax = fig.add_axes([0.91, 0.30, 0.015, 0.40])
    for i, hazard in enumerate(hazards_adv):
        for j, hour in enumerate(warning_hours):
            table = joint_status_table(data, hazard, hour)
            sns.heatmap(
                table,
                ax=axes[i, j],
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                vmin=0,
                vmax=0.5,
                cbar=(i == 0 and j == 0),
                cbar_ax=cbar_ax if (i == 0 and j == 0) else None,
                xticklabels=STATUS_LABELS,
                yticklabels=STATUS_LABELS,
                annot_kws={"fontsize": 9},
                linewidths=0.3,
                linecolor="white",
            )
            ax = axes[i, j]
            ax.set(xlabel="", ylabel="")
            ax.tick_params(axis="both", length=0, labelsize=9)
            ax.set_yticklabels(STATUS_LABELS, rotation=0)
            ax.set_xticklabels(STATUS_LABELS, rotation=0, ha="right")
            if i == 0:
                ax.set_title(f"{hour:+d} h" if hour else "0 h", fontsize=12, fontweight="bold")
            if j == 0:
                ax.annotate(labels_adv[i], xy=(-0.40, 0.5), xycoords="axes fraction", rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")
    fig.supxlabel(r"$\mathbf{CoU}$", fontsize=20, fontweight="bold", y=0.025)
    fig.supylabel(rf"$\mathbf{{DYN_{{{range_label}}}}}$", fontsize=20, fontweight="bold", x=0.015)
    cbar_ax.set_ylabel("Fraction of county-level cases", fontsize=11, labelpad=12)
    fig.suptitle("Joint Classification of Forecast Hazard Errors", fontsize=16, fontweight="bold", y=0.96)
    fig.savefig(FIGURE_DIR / f"dir_miscontainment_haz_adv_{dyn_range}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Individual rainfall heatmaps.
    for hazard, title in [("rain1", "Hourly Rain"), ("rainT", "Total Rain")]:
        fig, ax = plt.subplots()
        sns.heatmap(
            joint_status_table(data, hazard),
            ax=ax,
            annot=True,
            cmap="YlGnBu",
            vmin=0,
            vmax=0.5,
            cbar=True,
            xticklabels=STATUS_LABELS,
            yticklabels=STATUS_LABELS,
        )
        ax.tick_params(axis="both", labelsize=16, rotation=0)
        ax.set_ylabel(rf"$\mathbf{{DYN_{{{range_label}}}}}$", fontweight="bold", fontsize=15)
        ax.set_xlabel(r"$\mathbf{CoU}$", fontweight="bold", fontsize=15)
        ax.set_title(title, fontweight="bold", fontsize=20)
        fig.savefig(FIGURE_DIR / f"dir_miscontainment_haz_{hazard}_{dyn_range}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def rotate_gdf(gdf, angle=-30, origin=(-81.5, 27.5)):
    out = gdf.copy()
    out["geometry"] = out.geometry.apply(lambda geom: rotate(geom, angle, origin=origin))
    return out


def county_containment_stats(data):
    """Calculate county-level difference in DYN versus CoU containment frequency."""
    valid = data[data.dyn_status.notna() & data.nhc_status.notna()].copy()
    valid["dyn_contains"] = valid.dyn_status == 0
    valid["nhc_contains"] = valid.nhc_status == 0
    stats = (
        valid.groupby(["haz", "GEOID"], as_index=False)
        .agg(dynamic_contains=("dyn_contains", "mean"), nhc_contains=("nhc_contains", "mean"), n_cases=("dyn_contains", "size"))
    )
    stats["dynamic_advantage"] = stats.dynamic_contains - stats.nhc_contains
    return stats


def map_layers():
    """Load and rotate repository-relative map layers."""
    counties = gpd.read_file(COAST_DIR / "US_counties.shp")
    counties["GEOID"] = normalize_geoid(counties["GEOID"])
    states = rotate_gdf(gpd.read_file(COAST_DIR / "cb_2018_us_state_20m/cb_2018_us_state_20m.shp"))
    ocean = rotate_gdf(gpd.read_file(COAST_DIR / "ne_10m_ocean/ne_10m_ocean.shp"))
    lakes = rotate_gdf(gpd.read_file(COAST_DIR / "great_lakes/GL250515_lam.shp").to_crs(4326))
    return rotate_gdf(counties), states, ocean, lakes


def add_advantage_colorbar(fig, axes, label):
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-1, vmax=1), cmap="RdBu_r")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=np.asarray(axes).ravel().tolist(), orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label(label, fontsize=15)


def plot_containment_maps(data, dyn_range=DYN_RANGE):
    """Generate county maps of DYN minus CoU containment probability."""
    range_label = "F" if dyn_range == "full" else dyn_range
    stats = county_containment_stats(data)
    counties, states, ocean, lakes = map_layers()
    gdf = counties.merge(stats, on="GEOID", how="left")
    ocean_color = "#8FA6A9"

    def draw(ax, hazard):
        ax.set_facecolor("lightgrey")
        states.plot(ax=ax, edgecolor="k", facecolor="grey", linewidth=0.4, alpha=0.5)
        ocean.plot(ax=ax, facecolor=ocean_color, alpha=0.9)
        lakes.plot(ax=ax, facecolor=ocean_color, alpha=0.9)
        gdf[gdf["haz"] == hazard].plot(
            ax=ax,
            column="dynamic_advantage",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            legend=False,
            missing_kwds={"color": "lightgrey", "label": "Not evaluated"},
        )
        ax.set(xlim=(-99, -60), ylim=(24, 46))

    # 2x2 four-hazard layout.
    hazards = ["wind", "rain1", "surge", "power"]
    labels = ["Wind", "Hourly\nRain", "Storm\nTide", "Power\nOutage"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, hazard, label in zip(axes.flat, hazards, labels):
        draw(ax, hazard)
        ax.text(-70, 26, label, fontsize=15, fontweight="bold")
    fig.tight_layout()
    add_advantage_colorbar(fig, axes, rf"$\mathbf{{P_{{DYN_{{{range_label}}}}} - P_{{NHC}}}}$")
    fig.savefig(FIGURE_DIR / f"contain_prob_{dyn_range}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4x1 layout of the same four hazards.
    fig, axes = plt.subplots(4, 1, figsize=(5, 12))
    for ax, hazard, label in zip(axes, hazards, labels):
        draw(ax, hazard)
        ax.set_xticks([-95, -90, -85, -80, -75, -70, -65])
        ax.set_ylabel(label, fontsize=20, fontweight="bold")
    axes[0].set_title(rf"$\mathbf{{DYN_{{{range_label}}}}}$", fontsize=20, fontweight="bold")
    add_advantage_colorbar(fig, axes, r"$\mathbf{P_{DYN} - P_{NHC}}$")
    fig.savefig(FIGURE_DIR / f"contain_prob_{dyn_range}_1d.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Hourly versus total rainfall comparison.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, hazard, label in zip(axes, ["rain1", "rainT"], ["Hourly\nRain", "Total\nRain"]):
        draw(ax, hazard)
        ax.text(-70, 26, label, fontsize=15, fontweight="bold")
    fig.tight_layout()
    add_advantage_colorbar(fig, axes, rf"$\mathbf{{P_{{DYN_{{{range_label}}}}} - P_{{NHC}}}}$")
    fig.savefig(FIGURE_DIR / f"contain_prob_rain_{dyn_range}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    state = pd.read_csv(STATE_FILE)
    state["GEOID"] = normalize_geoid(state["GEOID"])
    plot_uncertainty_summary_figures(state)
    plot_reference_position_figures(state)
    plot_verification_figures(state)

    containment = prepare_containment(state, DYN_RANGE)
    plot_miscontainment_heatmaps(containment, DYN_RANGE)
    plot_containment_maps(containment, DYN_RANGE)


if __name__ == "__main__":
    main()

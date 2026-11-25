"""Build the county-level hazard-uncertainty table used by manuscript figures.

Output
------
postprocess/state_all_UP_2011_2024.csv

The plotting workflow in ``hazard_uncertainty_figures.py`` uses this single
processed table. The older notebooks also created separate extent/footprint
files, but those quantities are simple group summaries of this table and are
recomputed during plotting.
"""

import numpy as np
import pandas as pd

from config import (
    ADCIRC_DIR,
    HAZARD_COLUMNS,
    OBS_COLUMNS,
    POSTPROCESS_DIR,
    POWER_DIR,
    STORM_IDS_BY_YEAR,
    YEARS,
)
from data_utils import attach_power_columns

OUTPUT = POSTPROCESS_DIR / "state_all_UP_2011_2024.csv"
HAZARDS = ("wind", "rain1", "rain3", "rainT", "surge", "power")


def load_storm_forecasts(year, storm):
    """Load hazard summaries and attach power outputs by explicit row keys."""
    ofcl = pd.read_csv(ADCIRC_DIR / str(year) / f"county_summary_OFCL_{storm}.csv")
    dyn = pd.read_csv(ADCIRC_DIR / str(year) / f"county_summary_mod_{storm}.csv")
    ofcl_power = pd.read_csv(POWER_DIR / str(year) / f"county_summary_OFCL_{storm}.csv")
    dyn_power = pd.read_csv(POWER_DIR / str(year) / f"county_summary_mod_{storm}.csv")

    ofcl = attach_power_columns(
        ofcl,
        ofcl_power,
        [HAZARD_COLUMNS["power"]],
        year,
        storm,
        "OFCL",
    )
    dyn = attach_power_columns(
        dyn,
        dyn_power,
        [HAZARD_COLUMNS["power"], OBS_COLUMNS["power"]],
        year,
        storm,
        "dynamic",
    )

    merged = pd.concat([dyn, ofcl], ignore_index=True)
    merged.loc[merged["scounty_max"] < 0, "scounty_max"] = 0
    merged.loc[merged["scounty_max_obs"] < 0, "scounty_max_obs"] = 0

    # Preserve the model set used in the original analysis.
    dynamic_models = dyn["mods"].dropna().astype(str).str.strip().unique()
    return merged, dynamic_models


def summarize_hazard_uncertainty(
    forecasts,
    hazard,
    dynamic_models,
    threshold=0,
    model_col="mods",
    county_col="GEOID",
    warning_col="warning_hr",
):
    """Summarize CoU and dynamical hazard uncertainty by county/advisory.

    The dynamical representation includes the full min-max envelope and the
    central 67% interval (16.5th-83.5th percentiles). The CoU representation
    uses OFCI and the two cone scenarios. BEST-based observations are retained
    for containment and percentile-position analyses.
    """
    hazard_col = HAZARD_COLUMNS[hazard]
    obs_col = OBS_COLUMNS[hazard]
    df = forecasts.copy()
    df["_model"] = df[model_col].astype(str).str.strip()
    keys = [county_col, warning_col]

    # Dynamical distribution.
    dyn = df[df["_model"].isin(dynamic_models)].copy()
    grouped = dyn.groupby(keys)[hazard_col]
    dyn_stats = grouped.agg(
        dyn_mean="mean",
        dyn_median="median",
        dyn_std="std",
        dyn_min="min",
        dyn_max="max",
        n_dynamic="count",
    ).reset_index()
    quantiles = (
        grouped.quantile([0.165, 0.25, 0.75, 0.835])
        .unstack()
        .reset_index()
        .rename(columns={0.165: "dyn_p165", 0.25: "dyn_p25", 0.75: "dyn_p75", 0.835: "dyn_p835"})
    )
    dyn_stats = dyn_stats.merge(quantiles, on=keys, how="outer")

    # OFCL plus the two CoU scenarios.
    nhc = (
        df[df["_model"].isin(["OFCI", "OFCI_cone1", "OFCI_cone2"])]
        .pivot_table(index=keys, columns="_model", values=hazard_col, aggfunc="first")
        .reset_index()
        .rename(columns={"OFCI": "ofcl", "OFCI_cone1": "cone1", "OFCI_cone2": "cone2"})
    )
    for col in ("ofcl", "cone1", "cone2"):
        if col not in nhc:
            nhc[col] = np.nan

    nhc_values = nhc[["ofcl", "cone1", "cone2"]]
    nhc["n_nhc"] = nhc_values.notna().sum(axis=1)
    nhc["nhc_min"] = nhc_values.min(axis=1)
    nhc["nhc_max"] = nhc_values.max(axis=1)
    nhc.loc[nhc["n_nhc"] < 3, ["nhc_min", "nhc_max"]] = np.nan

    # BEST reference is constant by county in these county summary files.
    obs = (
        df[[county_col, obs_col]]
        .drop_duplicates(county_col)
        .rename(columns={obs_col: "obs"})
    )
    state = dyn_stats.merge(nhc, on=keys, how="outer").merge(obs, on=county_col, how="left")

    # Empirical BEST percentile within the dynamical members.
    rank_data = dyn[keys + [hazard_col]].merge(state[keys + ["obs"]], on=keys, how="left")
    rank_data["_below"] = (rank_data[hazard_col] < rank_data["obs"]).astype(float)
    rank_data["_equal"] = (rank_data[hazard_col] == rank_data["obs"]).astype(float)
    bad = rank_data[hazard_col].isna() | rank_data["obs"].isna()
    rank_data.loc[bad, ["_below", "_equal"]] = np.nan
    rank = rank_data.groupby(keys).agg(
        n_valid=("_below", "count"),
        n_below=("_below", "sum"),
        n_equal=("_equal", "sum"),
    ).reset_index()
    rank["best_dyn_percentile"] = np.where(
        rank["n_valid"] > 0,
        100 * (rank["n_below"] + 0.5 * rank["n_equal"]) / rank["n_valid"],
        np.nan,
    )
    state = state.merge(rank[keys + ["best_dyn_percentile"]], on=keys, how="left")

    # Width metrics.
    state["dyn_envelope_width"] = state["dyn_max"] - state["dyn_min"]
    state["dyn_width67"] = state["dyn_p835"] - state["dyn_p165"]
    state["nhc_envelope_width"] = state["nhc_max"] - state["nhc_min"]
    state["width_difference"] = state["dyn_envelope_width"] - state["nhc_envelope_width"]
    state["width_difference_norm"] = state["width_difference"] / (
        state["dyn_envelope_width"] + state["nhc_envelope_width"]
    ).replace(0, np.nan)
    state["width67_difference"] = state["dyn_width67"] - state["nhc_envelope_width"]
    state["width67_difference_norm"] = state["width67_difference"] / (
        state["dyn_width67"] + state["nhc_envelope_width"]
    ).replace(0, np.nan)

    # Possible/certain affected-county footprints.
    state["dyn_possible"] = state["dyn_max"] > threshold
    state["nhc_possible"] = state["nhc_max"] > threshold
    state["dyn_certain"] = state["dyn_min"] > threshold
    state["nhc_certain"] = state["nhc_min"] > threshold
    state["dyn67_possible"] = state["dyn_p835"] > threshold
    state["dyn67_certain"] = state["dyn_p165"] > threshold
    state["obs_affected"] = state["obs"] > threshold
    state["hazard"] = hazard
    return state


def build_state_table():
    """Process all manuscript storms and return one county-level table."""
    states = []
    for year in YEARS:
        for storm in STORM_IDS_BY_YEAR[year]:
            print(f"Processing {year} storm {storm}")
            forecasts, dynamic_models = load_storm_forecasts(year, storm)
            for hazard in HAZARDS:
                state = summarize_hazard_uncertainty(forecasts, hazard, dynamic_models)
                state["year"] = year
                state["tc_num"] = storm
                states.append(state)
    return pd.concat(states, ignore_index=True)


def main():
    state = build_state_table()
    state.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT} ({len(state):,} rows)")


if __name__ == "__main__":
    main()

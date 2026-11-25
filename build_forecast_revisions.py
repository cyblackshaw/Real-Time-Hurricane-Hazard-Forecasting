"""Build OFCI county-level forecast-revision metrics used by manuscript figures.

Output
------
postprocess/all_hazard_forecast_change.csv

Only OFCI is retained because every saved figure that uses this table filters to
OFCI. The original notebook also calculated unused dynamical-model statistics.
"""

import numpy as np
import pandas as pd

from config import HAZARD_COLUMNS, POWER_DIR, POSTPROCESS_DIR, STORM_IDS_BY_YEAR, YEARS

OUTPUT = POSTPROCESS_DIR / "all_hazard_forecast_change.csv"
WARNING_HOURS = np.arange(-72, 12, 6)  # -72 through +6 h
REVISION_HAZARDS = {
    hazard: HAZARD_COLUMNS[hazard]
    for hazard in ("wind", "rain1", "surge", "power")
}


def load_ofci_forecast(year, storm):
    """Load the combined county hazard/power table used for OFCI revisions.

    The power-system output already contains the wind, rainfall, storm-tide,
    and power-outage fields needed here, so no positional merge is required.
    """
    df = pd.read_csv(POWER_DIR / str(year) / f"county_summary_OFCL_{storm}.csv")
    required = {"GEOID", "warning_hr", "mods", *REVISION_HAZARDS.values()}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(
            f"Missing required columns in {year} storm {storm} OFCL power file: {missing}"
        )

    df.loc[df["scounty_max"] < 0, "scounty_max"] = 0
    return df


def calculate_forecast_change(df, year, storm, warning_hours=WARNING_HOURS):
    """Compare each OFCI advisory with the preceding 6-hour advisory.

    For each hazard, the normalized magnitude revision is

        sum(|H2 - H1|) / sum(max(H1, H2)).

    The function also calculates Jaccard footprint change and the balanced
    replacement fraction used in ``ofcl_hind_mae.png``.
    """
    rows = []
    ofci = df.loc[df["mods"].astype(str).str.strip() == "OFCI"].copy()
    ofci["GEOID"] = ofci["GEOID"].astype(str).str.zfill(5)
    ofci = ofci[ofci["warning_hr"].isin(warning_hours)]
    ofci = ofci.groupby(["GEOID", "warning_hr"], as_index=False).first()

    for hazard, column in REVISION_HAZARDS.items():
        for t0, t1 in zip(warning_hours[:-1], warning_hours[1:]):
            a = ofci.loc[ofci["warning_hr"] == t0].set_index("GEOID")[column]
            b = ofci.loc[ofci["warning_hr"] == t1].set_index("GEOID")[column]
            if a.empty or b.empty:
                continue

            counties = a.index.union(b.index)
            a, b = a.reindex(counties), b.reindex(counties)
            valid = np.isfinite(a) & np.isfinite(b)
            x = a[valid].to_numpy(dtype=float)
            y = b[valid].to_numpy(dtype=float)
            if not len(x):
                continue

            denom = np.maximum(x, y).sum()
            nmae = np.abs(y - x).sum() / denom if denom > 0 else 0.0

            # Threshold = 0, matching the original forecast-change analysis.
            f0, f1 = x > 0, y > 0
            intersection = np.sum(f0 & f1)
            union = np.sum(f0 | f1)
            jaccard_change = 1 - intersection / union if union else 0.0
            footprint_change = (f1.sum() - f0.sum()) / union if union else 0.0
            replacement = (
                np.clip(1 - abs(footprint_change) / jaccard_change, 0, 1)
                if jaccard_change > 0
                else np.nan
            )

            rows.append(
                {
                    "year": year,
                    "storm": storm,
                    "hazard": hazard,
                    "model": "OFCI",
                    "warning_hr_prev": t0,
                    "warning_hr": t1,
                    "n_counties": len(x),
                    "normalized_mae_change": nmae,
                    "footprint_change": footprint_change,
                    "jaccard_change": jaccard_change,
                    "replacement_fraction": replacement,
                }
            )

    return pd.DataFrame(rows)


def build_revision_table():
    tables = []
    for year in YEARS:
        for storm in STORM_IDS_BY_YEAR[year]:
            print(f"Processing {year} storm {storm}")
            tables.append(calculate_forecast_change(load_ofci_forecast(year, storm), year, storm))
    return pd.concat(tables, ignore_index=True)


def main():
    revisions = build_revision_table()
    revisions.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT} ({len(revisions):,} rows)")


if __name__ == "__main__":
    main()

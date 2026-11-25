"""Small data-alignment helpers shared by repository preprocessing scripts."""

import numpy as np
import pandas as pd

MERGE_KEYS = ("GEOID", "warning_hr", "mods")


def _normalize_merge_keys(df):
    """Normalize county, advisory, and model identifiers before merging."""
    out = df.copy()
    out["GEOID"] = (
        out["GEOID"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)
    )
    out["warning_hr"] = pd.to_numeric(out["warning_hr"], errors="coerce")
    out["mods"] = out["mods"].astype("string").str.strip()
    return out


def attach_power_columns(hazard, power, columns, year, storm, label):
    """Attach power outputs to hazard rows using explicit identifiers.

    The original notebooks assigned power columns by row position. Some storm
    files have different row counts, so positional assignment is unsafe. This
    routine matches rows by county, forecast lead time, and model aid instead.

    Duplicate power rows are allowed only when they contain identical requested
    values. Hazard rows without a matching power row are retained with NaN power
    values and reported so the mismatch can be inspected.
    """
    keys = list(MERGE_KEYS)
    required = keys + list(columns)
    missing_h = [c for c in keys if c not in hazard.columns]
    missing_p = [c for c in required if c not in power.columns]
    if missing_h or missing_p:
        raise KeyError(
            f"Cannot merge hazard/power data for {year} storm {storm} ({label}). "
            f"Missing hazard keys={missing_h}; missing power columns={missing_p}."
        )

    h = _normalize_merge_keys(hazard)
    p = _normalize_merge_keys(power)
    p = p[keys + list(columns)].copy()

    # A key may occur more than once in the power file, but repeated rows must
    # agree on the values being attached.
    conflicts = p.groupby(keys, dropna=False)[list(columns)].nunique(dropna=False)
    conflicts = conflicts[(conflicts > 1).any(axis=1)]
    if not conflicts.empty:
        sample = conflicts.head(5).index.tolist()
        raise ValueError(
            f"Conflicting duplicate power rows for {year} storm {storm} ({label}); "
            f"example keys: {sample}"
        )

    p = p.drop_duplicates(keys, keep="last")

    # Remove stale copies of the requested fields before attaching the keyed values.
    h = h.drop(columns=[c for c in columns if c in h.columns], errors="ignore")
    h["_row_order"] = np.arange(len(h))

    merged = h.merge(
        p,
        on=keys,
        how="left",
        validate="many_to_one",
        indicator="_power_merge",
        sort=False,
    )

    n_unmatched = int((merged["_power_merge"] == "left_only").sum())
    if n_unmatched:
        sample = (
            merged.loc[merged["_power_merge"] == "left_only", keys]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        print(
            f"WARNING: {year} storm {storm} ({label}): {n_unmatched}/{len(merged)} "
            f"hazard rows have no matching power row. Example keys: {sample}"
        )

    # Also report power records that have no corresponding hazard row.
    hazard_keys = h[keys].drop_duplicates()
    power_only = p[keys].drop_duplicates().merge(
        hazard_keys, on=keys, how="left", indicator=True
    )
    n_extra = int((power_only["_merge"] == "left_only").sum())
    if n_extra:
        print(
            f"WARNING: {year} storm {storm} ({label}): {n_extra} power keys "
            "have no corresponding hazard row and will be ignored."
        )

    return (
        merged.sort_values("_row_order")
        .drop(columns=["_row_order", "_power_merge"])
        .reset_index(drop=True)
    )

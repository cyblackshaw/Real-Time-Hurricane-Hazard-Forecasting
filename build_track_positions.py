"""Build paired BEST-position classifications for CoU and DYN track figures.

Output
------
postprocess/track_position_paired.csv

The original notebook retained many diagnostic quantities that were never used
by a saved figure. This version stores only the identifiers and categorical
positions required by ``track_figures.py``.
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from pyproj import CRS, Transformer

from config import ADECK_DIR, EXCLUDED_YEARS, POSTPROCESS_DIR, STORM_IDS_BY_YEAR, YEARS

OUTPUT = POSTPROCESS_DIR / "track_position_paired.csv"
MIN_WARNING, MAX_WARNING = -72, 0


def mat_2d(mat, key):
    """Extract a numeric MATLAB field as rows x forecast leads."""
    if key not in mat:
        raise KeyError(f"Missing MATLAB field: {key}")
    arr = np.squeeze(np.asarray(mat[key], dtype=float))
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"{key} must be 1D/2D after squeeze; got {arr.shape}.")
    return arr


def row_values(mat, key, n_rows):
    """Extract one numeric metadata value per MATLAB row."""
    arr = np.squeeze(np.asarray(mat[key], dtype=float))
    if arr.ndim == 0:
        return np.full(n_rows, float(arr))
    if arr.ndim == 1:
        if len(arr) == n_rows:
            return arr.astype(float)
        if n_rows == 1:
            finite = arr[np.isfinite(arr)]
            return np.array([finite[0] if finite.size else np.nan])
    if arr.ndim == 2:
        if arr.shape[1] == n_rows and arr.shape[0] != n_rows:
            arr = arr.T
        if arr.shape[0] == n_rows:
            return np.array([
                row[np.isfinite(row)][0] if np.isfinite(row).any() else np.nan
                for row in arr
            ])
    raise ValueError(f"Cannot map {key} shape {arr.shape} to {n_rows} rows.")


def observation_lookup(best):
    """Flatten BEST-track data for lookup by storm number and valid hour."""
    fields = {
        "lon": np.asarray(best["lonstore"], dtype=float).ravel(),
        "lat": np.asarray(best["latstore"], dtype=float).ravel(),
        "tc_num": np.asarray(best["tc_num"], dtype=float).ravel(),
        "warning_hr": np.asarray(best["warning_hr"], dtype=float).ravel(),
    }
    n = min(map(len, fields.values()))
    fields = {k: v[:n] for k, v in fields.items()}
    valid = np.logical_and.reduce([np.isfinite(v) for v in fields.values()])
    return {k: v[valid] for k, v in fields.items()}


def find_observation(obs, tc_num, valid_hour, tolerance=0.01):
    """Return BEST longitude/latitude for a storm and valid relative hour."""
    match = np.where(
        np.isclose(obs["tc_num"], tc_num, atol=tolerance, rtol=0)
        & np.isclose(obs["warning_hr"], valid_hour, atol=tolerance, rtol=0)
    )[0]
    if not match.size:
        return None
    i = match[0]
    return obs["lon"][i], obs["lat"][i]


def local_xy(lon, lat, center_lon, center_lat):
    """Project lon/lat to local azimuthal-equidistant x/y coordinates (km)."""
    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m"
    )
    transform = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transform.transform(lon, lat)
    return np.asarray(x) / 1000.0, np.asarray(y) / 1000.0


def normalized_cone_position(center, edge1, edge2, observed):
    """Return BEST cross-cone position (-1=edge1, 0=OFCL, +1=edge2)."""
    center_lon, center_lat = center
    values = np.asarray([*center, *edge1, *edge2, *observed], dtype=float)
    if not np.isfinite(values).all():
        return np.nan

    x, y = local_xy(
        [center_lon, edge1[0], edge2[0], observed[0]],
        [center_lat, edge1[1], edge2[1], observed[1]],
        center_lon,
        center_lat,
    )
    center_xy, e1, e2, obs_xy = map(np.asarray, zip(x, y))
    vector = e2 - e1
    width = np.linalg.norm(vector)
    if width <= 0:
        return np.nan

    unit = vector / width
    center_along = np.dot(center_xy - e1, unit)
    obs_from_center = np.dot(obs_xy - e1, unit) - center_along
    if obs_from_center < 0:
        return obs_from_center / center_along if center_along > 0 else np.nan
    right_width = width - center_along
    return obs_from_center / right_width if right_width > 0 else np.nan


def evaluate_cone_positions(ofcl, cone1, cone2, best, year, timestep=6):
    """Classify BEST as CoU side 1, CoU side 2, or outside for each forecast lead."""
    lon = mat_2d(ofcl, "lonstore")
    lat = mat_2d(ofcl, "latstore")
    lon1, lat1 = mat_2d(cone1, "lonstore"), mat_2d(cone1, "latstore")
    lon2, lat2 = mat_2d(cone2, "lonstore"), mat_2d(cone2, "latstore")
    if not (lon.shape == lat.shape == lon1.shape == lat1.shape == lon2.shape == lat2.shape):
        raise ValueError("OFCL and cone arrays must have matching shapes.")

    n_rows, n_leads = lon.shape
    tc_nums = row_values(ofcl, "tc_num", n_rows)
    warnings = row_values(ofcl, "warning_hr", n_rows)
    obs = observation_lookup(best)
    rows = []

    for i in range(n_rows):
        tc, warning = tc_nums[i], warnings[i]
        if not np.isfinite(tc) or not (MIN_WARNING <= warning <= MAX_WARNING):
            continue
        for j in range(n_leads):
            lead = j * timestep
            observed = find_observation(obs, tc, warning + lead)
            if observed is None:
                continue
            pos = normalized_cone_position(
                (lon[i, j], lat[i, j]),
                (lon1[i, j], lat1[i, j]),
                (lon2[i, j], lat2[i, j]),
                observed,
            )
            if not np.isfinite(pos):
                continue
            category = "cone1" if -1 <= pos < 0 else "cone2" if 0 <= pos <= 1 else "outside"
            rows.append(
                {
                    "year": year,
                    "tc_num": tc,
                    "warning_hr": warning,
                    "lead_hour": lead,
                    "cone_position": pos,
                    "cone_position_class": category,
                }
            )
    return pd.DataFrame(rows)


def mean_knn_distance(reference, values, k):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(reference) or len(values) < k:
        return np.nan
    distances = np.abs(values - reference)
    return float(np.partition(distances, k - 1)[:k].mean())


def member_knn_distances(values, k):
    """Mean distance from each member to its k nearest other members."""
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) <= k:
        return out
    for i, value in enumerate(values):
        distances = np.abs(values - value)
        distances[i] = np.inf
        finite = distances[np.isfinite(distances)]
        if len(finite) >= k:
            out[i] = np.partition(finite, k - 1)[:k].mean()
    return out


def evaluate_dynamic_positions(
    dyn,
    ofcl,
    best,
    year,
    k=2,
    dense_fraction=0.67,
    minimum_models=4,
    timestep=6,
):
    """Classify BEST as dense, sparse, or outside the DYN cross-track spread."""
    dyn_lon, dyn_lat = mat_2d(dyn, "lonstore"), mat_2d(dyn, "latstore")
    ofcl_lon, ofcl_lat = mat_2d(ofcl, "lonstore"), mat_2d(ofcl, "latstore")
    if dyn_lon.shape != dyn_lat.shape or ofcl_lon.shape != ofcl_lat.shape:
        raise ValueError("Longitude/latitude array shapes do not match.")

    n_dyn, n_dyn_leads = dyn_lon.shape
    n_ofcl, n_ofcl_leads = ofcl_lon.shape
    n_leads = min(n_dyn_leads, n_ofcl_leads)
    dyn_tc = row_values(dyn, "tc_num", n_dyn)
    dyn_warning = row_values(dyn, "warning_hr", n_dyn)
    ofcl_tc = row_values(ofcl, "tc_num", n_ofcl)
    ofcl_warning = row_values(ofcl, "warning_hr", n_ofcl)
    obs = observation_lookup(best)
    required = max(minimum_models, k + 1)
    rows = []

    for oi in range(n_ofcl):
        tc, warning = ofcl_tc[oi], ofcl_warning[oi]
        if not np.isfinite(tc) or not (MIN_WARNING <= warning <= MAX_WARNING):
            continue
        dyn_rows = np.where(
            np.isclose(dyn_tc, tc, atol=0.01, rtol=0)
            & np.isclose(dyn_warning, warning, atol=0.01, rtol=0)
        )[0]
        if len(dyn_rows) < required:
            continue

        lats = dyn_lat[dyn_rows, :n_leads]
        lons = dyn_lon[dyn_rows, :n_leads]
        mean_lat = np.full(n_leads, np.nan)
        mean_lon = np.full(n_leads, np.nan)
        for j in range(n_leads):
            valid = np.isfinite(lats[:, j]) & np.isfinite(lons[:, j])
            if valid.sum() >= required:
                mean_lat[j] = lats[valid, j].mean()
                mean_lon[j] = lons[valid, j].mean()

        for j in range(n_leads):
            center = (mean_lon[j], mean_lat[j])
            if not np.isfinite(center).all():
                continue
            valid = np.isfinite(lats[:, j]) & np.isfinite(lons[:, j])
            if valid.sum() < required:
                continue

            observed = find_observation(obs, tc, warning + j * timestep)
            if observed is None:
                continue

            # Local trajectory direction of the ensemble-mean track.
            if 0 < j < n_leads - 1 and np.isfinite(
                [mean_lon[j - 1], mean_lat[j - 1], mean_lon[j + 1], mean_lat[j + 1]]
            ).all():
                px, py = local_xy(mean_lon[j - 1], mean_lat[j - 1], *center)
                nx, ny = local_xy(mean_lon[j + 1], mean_lat[j + 1], *center)
                dx, dy = nx - px, ny - py
            elif j < n_leads - 1 and np.isfinite([mean_lon[j + 1], mean_lat[j + 1]]).all():
                dx, dy = local_xy(mean_lon[j + 1], mean_lat[j + 1], *center)
            elif j > 0 and np.isfinite([mean_lon[j - 1], mean_lat[j - 1]]).all():
                px, py = local_xy(mean_lon[j - 1], mean_lat[j - 1], *center)
                dx, dy = -px, -py
            else:
                continue

            length = np.hypot(dx, dy)
            if not np.isfinite(length) or length <= 0:
                continue
            cross_unit = np.array([-dy / length, dx / length])

            model_x, model_y = local_xy(lons[valid, j], lats[valid, j], *center)
            best_x, best_y = local_xy(observed[0], observed[1], *center)
            model_cross = np.column_stack([model_x, model_y]) @ cross_unit
            best_cross = np.array([best_x, best_y]) @ cross_unit

            contained = model_cross.min() <= best_cross <= model_cross.max()
            member_knn = member_knn_distances(model_cross, k)
            member_knn = member_knn[np.isfinite(member_knn)]
            if not len(member_knn):
                continue
            threshold = np.quantile(member_knn, dense_fraction)
            best_knn = mean_knn_distance(best_cross, model_cross, k)

            if not contained:
                position = "outside"
            elif np.isfinite(best_knn) and best_knn <= threshold:
                position = "dense"
            else:
                position = "sparse"

            rows.append(
                {
                    "year": year,
                    "tc_num": tc,
                    "warning_hr": warning,
                    "lead_hour": j * timestep,
                    "best_position": position,
                }
            )
    return pd.DataFrame(rows)


def load_year(year):
    """Load the five MAT files needed for track-position analysis."""
    return {
        "ofcl": sio.loadmat(ADECK_DIR / f"{year}_land_ofcl_tracks_3day_FULL.mat"),
        "cone1": sio.loadmat(ADECK_DIR / f"{year}_land_ofcl_tracks_3day_cone1_FULL.mat"),
        "cone2": sio.loadmat(ADECK_DIR / f"{year}_land_ofcl_tracks_3day_cone2_FULL.mat"),
        "best": sio.loadmat(ADECK_DIR / f"{year}_land_obs_tracks_3day.mat"),
        "dyn": sio.loadmat(ADECK_DIR / f"{year}_land_era_mod_3day_tracks_FULL.mat"),
    }


def build_paired_positions():
    """Evaluate all years and retain cases available to both CoU and DYN."""
    paired = []
    keys = ["year", "tc_num", "warning_hr", "lead_hour"]
    for year in YEARS:
        if year in EXCLUDED_YEARS:
            continue
        print(f"Processing {year}")
        data = load_year(year)
        cone = evaluate_cone_positions(data["ofcl"], data["cone1"], data["cone2"], data["best"], year)
        dyn = evaluate_dynamic_positions(data["dyn"], data["ofcl"], data["best"], year)
        year_paired = cone.merge(dyn, on=keys, how="inner")

        # Retain only the manuscript storms listed in the shared configuration.
        storm_ids = {float(storm) for storm in STORM_IDS_BY_YEAR[year]}
        year_paired = year_paired[year_paired["tc_num"].isin(storm_ids)]
        paired.append(year_paired)
    return pd.concat(paired, ignore_index=True)


def main():
    paired = build_paired_positions()
    paired.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT} ({len(paired):,} paired cases)")


if __name__ == "__main__":
    main()

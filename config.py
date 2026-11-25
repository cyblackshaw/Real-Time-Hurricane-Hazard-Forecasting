"""Shared configuration for manuscript figure-generation scripts."""

from pathlib import Path

YEARS = tuple(range(2011, 2025))
EXCLUDED_YEARS = {2013, 2015}

STORM_IDS_BY_YEAR = {
    2011: [9],
    2012: [9, 18],
    2013: [],
    2014: [1],
    2015: [],
    2016: [9],
    2017: [9, 11, 16],
    2018: [6, 14],
    2019: [2],
    2020: [8, 9, 13, 19, 26, 28],
    2021: [9, 14],
    2022: [9, 17],
    2023: [10],
    2024: [2, 4, 6, 9, 14],
}

HAZARD_COLUMNS = {
    "wind": "wcounty_max",
    "rain1": "pcounty_max1",
    "rain3": "pcounty_max3",
    "rainT": "pcounty_max",
    "surge": "scounty_max",
    "power": "pred_peak_customers_affected_overtopping",
}

OBS_COLUMNS = {
    "wind": "wcounty_max_obs",
    "rain1": "pcounty_max1_obs",
    "rain3": "pcounty_max3_obs",
    "rainT": "pcounty_max_obs",
    "surge": "scounty_max_obs",
    "power": "pred_peak_customers_affected_christine_best_overtopping",
}

HAZARD_LABELS = {
    "wind": "Wind",
    "rain1": "Hourly Rain",
    "rain3": "3-Hourly Rain",
    "rainT": "Total Rain",
    "surge": "Storm Tide",
    "power": "Power Outage",
}

# Repository-relative paths. Change these here rather than inside figure code.
FIGURE_DIR = Path("Figures")
POSTPROCESS_DIR = Path("postprocess")
ADECK_DIR = Path("adeck")
BDECK_DIR = Path("bdeck")
ADCIRC_DIR = Path("model_ouputs")
POWER_DIR = Path("model_ouputs/with_power_system")
COAST_DIR = Path("coast")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
POSTPROCESS_DIR.mkdir(parents=True, exist_ok=True)

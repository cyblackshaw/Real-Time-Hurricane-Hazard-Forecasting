# Tropical Cyclone Multi-Hazard Forecast Uncertainty

Code and processed data used to reproduce the analyses and figures for a manuscript evaluating uncertainty in tropical-cyclone track and multi-hazard forecasts across United States landfalling storms from 2011–2024.

The workflow compares uncertainty represented by the National Hurricane Center (NHC) forecast track and Cone of Uncertainty with uncertainty represented by dynamical-model guidance, and evaluates how those differences propagate into forecasts of wind, rainfall, storm tide, and power outages.

This repository is a condensed, publication-facing version of the exploratory analysis notebooks. It retains the calculations needed for manuscript figures while removing notebook diagnostics, duplicate code, machine-specific paths, and analyses that were not used in saved figures.

## Repository structure

```text
.
├── README.md
├── code/
│   ├── config.py
│   ├── data_utils.py
│   ├── build_track_positions.py
│   ├── build_forecast_revisions.py
│   ├── build_hazard_state.py
│   ├── track_figures.py
│   ├── forecast_revision_figures.py
│   ├── hazard_uncertainty_figures.py
│   ├── make_figures.py
│   └── requirements.txt
├── postprocess/
│   ├── track_position_paired.csv
│   ├── all_hazard_forecast_change.csv
│   └── state_all_UP_2011_2024.csv
└── Figures/
```

The three files in `postprocess/` are the primary processed inputs used by the manuscript figure scripts. They are included so that the published figures can be reproduced without rerunning the full upstream hazard-model workflow.

Raw and intermediate model files are not distributed with this repository.

## Study storms

The storm sample is defined in `code/config.py` and should be treated as the authoritative list used by the scripts.

| Year | Atlantic storm numbers |
|---|---|
| 2011 | 09 |
| 2012 | 09, 18 |
| 2013 | excluded |
| 2014 | 01 |
| 2015 | excluded |
| 2016 | 09 |
| 2017 | 09, 11, 16 |
| 2018 | 06, 14 |
| 2019 | 02 |
| 2020 | 08, 09, 13, 19, 26, 28 |
| 2021 | 09, 14 |
| 2022 | 09, 17 |
| 2023 | 10 |
| 2024 | 02, 04, 06, 09, 14 |

## Analysis streams

### 1. Track uncertainty

`build_track_positions.py` creates:

```text
postprocess/track_position_paired.csv
```

The processed table pairs BEST-track positions evaluated relative to:

- the NHC official forecast track and Cone of Uncertainty; and
- the cross-track distribution of dynamical-model guidance.

`track_figures.py` uses the processed table to create the paired track-position figures and also creates the observed-storm-track map and conceptual track schematic.

The upstream track analysis uses NHC Automated Tropical Cyclone Forecasting System (ATCF) A-deck and B-deck records. Raw ATCF files are not included in this repository because they are publicly available from NHC.

NHC ATCF archive:

https://ftp.nhc.noaa.gov/atcf/archive/

The archive is organized by year. For Atlantic tropical cyclones, filenames generally follow:

```text
aalNNYYYY.dat.gz   # A-deck
balNNYYYY.dat.gz   # B-deck
```

where `NN` is the storm number and `YYYY` is the year. For example, Atlantic storm 09 in 2011 is represented by `aal092011.dat.gz` and `bal092011.dat.gz`.

The MATLAB files read by `build_track_positions.py` are derived preprocessing products from these ATCF records and are not redistributed here. Therefore, `track_position_paired.csv` should be used as the direct reproducibility input for the manuscript track-position figures. The builder is retained to document the analysis applied to those derived track files.

The observed-track map in `track_figures.py` expects locally prepared B-deck CSV files with paths of the form:

```text
bdeck/<year>/alNNYYYY.csv
```

These CSVs are also not distributed. Users wishing to regenerate the storm-track map should obtain the corresponding B-deck records from the NHC ATCF archive and convert them to the local CSV format expected by the plotting script.

### 2. Forecast revisions

`build_forecast_revisions.py` creates:

```text
postprocess/all_hazard_forecast_change.csv
```

This analysis quantifies changes between successive 6-hour OFCI forecast cycles for:

- wind;
- hourly rainfall;
- storm tide; and
- power outages.

For each hazard, the processed table contains normalized magnitude changes and footprint-change metrics used by the manuscript forecast-revision figures.

The builder reads the combined county hazard/power-output files from:

```text
inputs/with_power_system/<year>/county_summary_OFCL_<storm>.csv
```

These model-output files are not included in the repository. The archived processed table is therefore the recommended input for reproducing the manuscript figures.

### 3. Hazard uncertainty and verification

`build_hazard_state.py` creates:

```text
postprocess/state_all_UP_2011_2024.csv
```

This table summarizes county-level hazard uncertainty for:

- wind;
- hourly rainfall;
- 3-hour rainfall;
- total rainfall;
- storm tide; and
- power outages.

The analysis retains the NHC-based uncertainty representation (`OFCI`, `OFCI_cone1`, and `OFCI_cone2`), the dynamical-model distribution, and the BEST-based reference values used for containment and percentile-position analyses.

The builder reads county-level hazard files from:

```text
inputs/<year>/county_summary_OFCL_<storm>.csv
inputs/<year>/county_summary_mod_<storm>.csv
```

and power-system files from:

```text
inputs/with_power_system/<year>/county_summary_OFCL_<storm>.csv
inputs/with_power_system/<year>/county_summary_mod_<storm>.csv
```

The raw/model-output files are not redistributed. `postprocess/state_all_UP_2011_2024.csv` is provided as the direct input to the manuscript hazard-uncertainty and verification figures.

## Processed data products

| File | Purpose |
|---|---|
| `postprocess/track_position_paired.csv` | Paired BEST-track classifications for the NHC Cone of Uncertainty and dynamical-model track distribution |
| `postprocess/all_hazard_forecast_change.csv` | County-level changes between successive OFCI hazard forecasts |
| `postprocess/state_all_UP_2011_2024.csv` | County/advisory hazard-uncertainty, containment, percentile, and verification metrics |

These processed tables are the preferred starting point for manuscript figure reproduction. The builder scripts are retained for provenance and for users who have access to the corresponding upstream model outputs.

## Software requirements

The analysis was written in Python. Required packages are listed in:

```text
code/requirements.txt
```

A minimal environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r code/requirements.txt
```

The current requirements are:

```text
numpy
pandas
matplotlib
scipy
seaborn
geopandas
shapely
pyproj
```

## Configuration

Shared analysis settings are defined in `code/config.py`, including:

- analysis years;
- storm numbers by year;
- hazard and reference column names;
- display labels; and
- repository-relative paths.

The current relative paths are:

```python
FIGURE_DIR = Path("Figures")
POSTPROCESS_DIR = Path("postprocess")
ADECK_DIR = Path("adeck")
BDECK_DIR = Path("bdeck")
ADCIRC_DIR = Path("inputs")
POWER_DIR = Path("inputs/with_power_system")
COAST_DIR = Path("coast")
```

If the repository is reorganized, update these locations in `config.py` rather than editing individual analysis or plotting scripts.

## Reproducing manuscript figures

### From the processed tables

Place the three processed CSV files in `postprocess/`, then run from the repository root:

```bash
python code/make_figures.py
```

Figures are written to:

```text
Figures/
```

`make_figures.py` calls the track, forecast-revision, and hazard-uncertainty plotting modules in sequence.

Some map-based figures also require external geographic or track files described below. If those files are not available, run the relevant figure modules individually after providing their required inputs.

### Rebuilding the processed tables

If the upstream model products are available, rebuild the intermediate tables with:

```bash
python code/build_track_positions.py
python code/build_forecast_revisions.py
python code/build_hazard_state.py
```

Then regenerate the figures:

```bash
python code/make_figures.py
```

The builders are substantially lighter than the original modeling workflow. They begin from already-generated track and county-level model outputs rather than rerunning the underlying tropical-cyclone, rainfall, storm-tide, or power-outage models.

## External inputs not distributed with this repository

### NHC A-deck and B-deck data

Official NHC ATCF records can be downloaded from:

https://ftp.nhc.noaa.gov/atcf/archive/

NHC documentation and verification information describing the A-deck and B-deck databases are available at:

https://www.nhc.noaa.gov/verification/verify2.shtml

Only the storms listed in `code/config.py` are required.

### County hazard and power-system model outputs

The county-level files used by `build_forecast_revisions.py` and `build_hazard_state.py` are outputs from the manuscript modeling workflow and are not distributed as raw inputs here. The derived processed tables needed for the manuscript figures are included instead.

### Geographic layers

Several map-based figures use locally stored geographic layers under `coast/`. The current plotting code expects:

```text
coast/US_counties.shp
coast/cb_2018_us_state_20m/cb_2018_us_state_20m.shp
coast/ne_10m_ocean/ne_10m_ocean.shp
coast/great_lakes/GL250515_lam.shp
```

These layers are not required for the non-map statistical figures. If equivalent geographic files are used, update `COAST_DIR` and/or the relevant filenames in the plotting code.

## Figure outputs

The code generates the saved manuscript figures associated with the three analysis streams. Examples include:

```text
Figures/ofcl_hind_tracks.png
Figures/dyn_hind_tracks.png
Figures/storm_tracks.png
Figures/hind_tracks_schematics.png
Figures/ofcl_hind_mae.png
Figures/ofcl_hind_peak_timing.png
Figures/ofcl_hind_peak_timing_distribution.png
Figures/ofcl_peak_timing_heatmap.png
Figures/spearman_storm_haz.png
Figures/timing_storm_haz.png
Figures/mean_haz_change_forecast.png
Figures/haz_width_diff_w67.png
Figures/haz_extent_diff_w67.png
Figures/haz_jacc_sim_w67.png
Figures/fraction_contain_haz.png
Figures/reference_position_percentile.png
Figures/dyn_above_reference_percentile.png
```

Additional verification, directional-miscontainment, and county containment-probability figures are generated by `hazard_uncertainty_figures.py`.

## Notes on reproducibility

### Processed tables are intentional repository products

The full hazard-model workflow includes large model outputs and machine-specific/HPC processing that are not necessary to reproduce the manuscript figures. The three processed tables in `postprocess/` are therefore archived intentionally as stable analysis products.

This design provides two levels of reproducibility:

1. **Figure reproducibility:** use the archived processed tables to regenerate the manuscript figures.
2. **Analysis provenance:** use the builder scripts to document how those processed tables were constructed from upstream track and model-output files.

### Power-output alignment

The exploratory notebooks originally attached some power-outage values to hazard tables by row position. Because row counts are not identical for every storm, the repository version does not rely on positional alignment.

`build_hazard_state.py` instead joins power values using explicit county/advisory/model identifiers. `build_forecast_revisions.py` reads the combined power-system county file directly because that file already contains the hazard fields used in the revision analysis.

### Track preprocessing

`build_track_positions.py` begins from derived MATLAB track files rather than directly parsing raw ATCF text files. Those MATLAB files are not distributed in this repository. The raw source records can be obtained from the NHC ATCF archive, while `postprocess/track_position_paired.csv` provides the exact processed track-position input used by the manuscript figure code.

## Code organization

The publication-facing scripts were condensed from the exploratory notebooks with the following principles:

- retain calculations that contribute to saved manuscript figures;
- remove notebook-only diagnostics and unsaved exploratory plots;
- consolidate repeated utilities and configuration;
- replace machine-specific absolute paths with repository-relative paths;
- preserve analysis definitions used by the manuscript; and
- archive processed figure inputs when rerunning the full upstream modeling workflow would be unnecessarily burdensome.

## Citation

If you use this code or the processed data, please cite the associated manuscript and archived repository DOI.

Repository DOI: **[add DOI after archive publication]**

Manuscript citation: **[add final manuscript citation]**

## Contact

Questions about the analysis or repository can be directed to the corresponding author through the contact information provided with the manuscript.

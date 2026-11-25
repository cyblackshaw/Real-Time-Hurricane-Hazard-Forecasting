"""Generate all manuscript figures from processed analysis tables."""

from forecast_revision_figures import main as make_revision_figures
from hazard_uncertainty_figures import main as make_hazard_figures
from track_figures import main as make_track_figures


def main():
    make_track_figures()
    make_revision_figures()
    make_hazard_figures()


if __name__ == "__main__":
    main()

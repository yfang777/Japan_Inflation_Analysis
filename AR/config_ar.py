"""AR/config_ar.py — constants for the univariate AR family on Japan CPI."""

from pathlib import Path

ROOT  = Path(__file__).parent
PLOTS = ROOT / 'plots'
PLOTS.mkdir(exist_ok=True)

# horizons (months ahead) — see plan.md Phase 0
HORIZONS = [1, 3, 6, 12, 24]

# number of own lags
N_LAGS = 12

# evaluation windows (decision-date indexed)
WINDOWS = [
    ('Window A (2010m1-2019m12)', '2010-01-01', '2019-12-31'),
    ('Window B (2020m1-2025m12)', '2020-01-01', '2025-12-31'),
]

# expanding window starts here
TRAIN_START = '1995-01-01'

# minimum training observations
MIN_TRAIN = 120

# Japan consumption-tax hike impact months
TAX_HIKE_DATES = ['1997-04-01', '2014-04-01', '2019-10-01']

# subsample boundaries for Phase 5
SUBSAMPLES = [
    ('1995-2012 (deflation)',     '1995-01-01', '2012-12-31'),
    ('2013-2025 (post-Abenomics)', '2013-01-01', '2025-12-31'),
]

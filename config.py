"""
config.py  –  static knowledge about Japan CPI data structure

All column classifications, basket mappings, and display settings live here.
The analysis script imports from this file so structural changes stay in one place.
"""

from pathlib import Path
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_FILE   = ROOT / 'data_clean' / 'japan_inflation_structured_check.csv'
LEVEL_DIR   = ROOT / 'data_clean'          # level_1.csv, level_2.csv, level_3.csv
WEIGHTS_CSV = ROOT / 'data_clean' / 'official_weights_national.csv'
PLOTS_DIR   = ROOT / 'plots'

# ── composite / derived measures ───────────────────────────────────────────────
# These are computed FROM the true components. Never use as regression features —
# it would be regressing headline CPI on near-headline CPI.
COMPOSITE_COLS = [
    'All items',
    'All items, less fresh food',
    'All items, less imputed rent',
    'All items, less imputed rent & fresh food',
    'All items, less fresh food and energy',
    'All items, less food (less alcoholic beverages) and energy',
]

# ── special / 別掲 cross-cutting aggregates ────────────────────────────────────
# These overlap with the true components (e.g. Energy = Electricity + Gas +
# Other fuel + part of Private transportation). Including both the aggregate
# and its constituent components double-counts that slice of the basket.
SPECIAL_COLS = [
    'Energy',
    'Expenses for education',
    'Expenses for culture & recreation',
    'Expenses for information & communi-cation',
]

# ── isolated components ────────────────────────────────────────────────────────
# Components excluded from the main regression feature set for specific reasons.
# Each entry documents why and what the correct treatment is.
ISOLATED = {
    'Rent': {
        'basket_share':  584_936_605 / 3_190_396_706,   # 18.33%
        'market_share':   80_947_671 / 3_190_396_706,   #  2.54%  (持家の帰属家賃を除く家賃)
        'imputed_share': 503_988_934 / 3_190_396_706,   # 15.79%  (derived)
        'sigma_3m3m':    1.09,
        'reason': (
            'Dominated by imputed rent (持家の帰属家賃, ~85% of series). '
            'Imputed rent is an administrative estimate for owner-occupiers; '
            'it barely changes in Japan. σ(3m/3m) = 1.09 — lowest of all '
            'components — making it a weight sink in constrained regression. '
            'No separate market-rent series available in this dataset.'
        ),
    }
}

# ── true basket components (46) ────────────────────────────────────────────────
# Non-overlapping, sum to 81.67% of the basket (100% minus Rent's 18.33%).
# These are the correct regression features.
COMPONENT_COLS = [
    # ── Food ──────────────────────────────────────────────────────────────────
    'Cereals',
    'Fish & seafood',
    'Meats',
    'Dairy products & eggs',
    'Vegetables & seaweeds',
    'Fruits',
    'Oils, fats & seasonings',
    'Cakes & candies',
    'Cooked food',
    'Beverages',
    'Alcoholic beverages',
    'Meals outside the home',
    # ── Housing (ex Rent) ─────────────────────────────────────────────────────
    'Repairs & maintenance',
    # ── Utilities ─────────────────────────────────────────────────────────────
    'Electric-ity',
    'Gas',
    'Other fuel & light',
    'Water & sewerage charges',
    # ── Household goods & services ────────────────────────────────────────────
    'Household durable goods',
    'Interior furnishings',
    'Bedding',
    'Domestic utensils',
    'Domestic non- durable goods',
    'Domestic services',
    # ── Clothing ──────────────────────────────────────────────────────────────
    'Clothes',
    'Shirts, sweaters & underwear',
    'Footwear',
    'Other clothing',
    'Services related to clothing',
    # ── Healthcare ────────────────────────────────────────────────────────────
    'Medicines & health for- tification',
    'Medical supplies & appliances',
    'Medical services',
    # ── Transport & communication ─────────────────────────────────────────────
    'Public transpor- tation',
    'Private transpor- tation',
    'Communi- cation',
    # ── Education ─────────────────────────────────────────────────────────────
    'School fees',
    'School textbooks & reference books for study',
    'Tutorial fees',
    # ── Recreation ────────────────────────────────────────────────────────────
    'Recre- ational durable goods',
    'Recre- ational goods',
    'Books & other reading materials',
    'Recre- ational services',
    # ── Miscellaneous ─────────────────────────────────────────────────────────
    'Personal care services',
    'Toilet articles',
    'Personal effects',
    'Tobacco',
    'Other miscella- neous',
]

# ── English → Japanese for official weight lookup ──────────────────────────────
EN_TO_JPN = {
    'Cereals':                                    '穀類',
    'Fish & seafood':                             '魚介類',
    'Meats':                                      '肉類',
    'Dairy products & eggs':                      '乳卵類',
    'Vegetables & seaweeds':                      '野菜・海藻',
    'Fruits':                                     '果物',
    'Oils, fats & seasonings':                    '油脂・調味料',
    'Cakes & candies':                            '菓子類',
    'Cooked food':                                '調理食品',
    'Beverages':                                  '飲料',
    'Alcoholic beverages':                        '酒類',
    'Meals outside the home':                     '外食',
    'Rent':                                       '家賃',
    'Repairs & maintenance':                      '設備修繕・維持',
    'Electric-ity':                               '電気代',
    'Gas':                                        'ガス代',
    'Other fuel & light':                         '他の光熱',
    'Water & sewerage charges':                   '上下水道料',
    'Household durable goods':                    '家庭用耐久財',
    'Interior furnishings':                       '室内装備品',
    'Bedding':                                    '寝具類',
    'Domestic utensils':                          '家事雑貨',
    'Domestic non- durable goods':                '家事用消耗品',
    'Domestic services':                          '家事サービス',
    'Clothes':                                    '衣料',
    'Shirts, sweaters & underwear':               'シャツ・セーター・下着類',
    'Footwear':                                   '履物類',
    'Other clothing':                             '他の被服',
    'Services related to clothing':               '被服関連サービス',
    'Medicines & health for- tification':         '医薬品・健康保持用摂取品',
    'Medical supplies & appliances':              '保健医療用品・器具',
    'Medical services':                           '保健医療サービス',
    'Public transpor- tation':                    '交通',
    'Private transpor- tation':                   '自動車等関係費',
    'Communi- cation':                            '通信',
    'School fees':                                '授業料等',
    'School textbooks & reference books for study': '教科書・学習参考教材',
    'Tutorial fees':                              '補習教育',
    'Recre- ational durable goods':               '教養娯楽用耐久財',
    'Recre- ational goods':                       '教養娯楽用品',
    'Books & other reading materials':            '書籍・他の印刷物',
    'Recre- ational services':                    '教養娯楽サービス',
    'Personal care services':                     '理美容サービス',
    'Toilet articles':                            '理美容用品',
    'Personal effects':                           '身の回り用品',
    'Tobacco':                                    'たばこ',
    'Other miscella- neous':                      '他の諸雑費',
}

# ── groups for contribution / basket charts ────────────────────────────────────
# Rent is absent here; it is drawn separately in every chart.
GROUPS = {
    'Fresh food':      ['Vegetables & seaweeds', 'Fruits', 'Fish & seafood'],
    'Food (other)':    ['Cereals', 'Meats', 'Dairy products & eggs',
                        'Oils, fats & seasonings', 'Cakes & candies',
                        'Cooked food', 'Beverages', 'Alcoholic beverages',
                        'Meals outside the home'],
    'Energy':          ['Electric-ity', 'Gas', 'Other fuel & light'],
    'Housing (other)': ['Repairs & maintenance', 'Water & sewerage charges'],
    'Transport':       ['Public transpor- tation', 'Private transpor- tation'],
    'Communication':   ['Communi- cation'],
    'Household goods': ['Household durable goods', 'Interior furnishings', 'Bedding',
                        'Domestic utensils', 'Domestic non- durable goods',
                        'Domestic services'],
    'Clothing':        ['Clothes', 'Shirts, sweaters & underwear', 'Footwear',
                        'Other clothing', 'Services related to clothing'],
    'Healthcare':      ['Medicines & health for- tification',
                        'Medical supplies & appliances', 'Medical services'],
    'Education':       ['School fees',
                        'School textbooks & reference books for study', 'Tutorial fees'],
    'Recreation':      ['Recre- ational durable goods', 'Recre- ational goods',
                        'Books & other reading materials', 'Recre- ational services'],
    'Other':           ['Personal care services', 'Toilet articles', 'Personal effects',
                        'Tobacco', 'Other miscella- neous'],
}

GROUP_COLORS = {
    'Fresh food':      '#27ae60',
    'Food (other)':    '#2ecc71',
    'Energy':          '#e74c3c',
    'Housing (other)': '#9b59b6',
    'Transport':       '#e67e22',
    'Communication':   '#f39c12',
    'Household goods': '#bdc3c7',
    'Clothing':        '#1abc9c',
    'Healthcare':      '#3498db',
    'Education':       '#2980b9',
    'Recreation':      '#16a085',
    'Other':           '#7f8c8d',
    'Rent':            '#8e44ad',   # for isolated display only
}

# ── derived lookups ──────────────────────────────────────────────────────────
COL_TO_GROUP = {col: grp for grp, cols in GROUPS.items() for col in cols}

# ── regression feature set ───────────────────────────────────────────────────
# 46 clean components + Rent (isolated but included for comparison)
FEATURES = COMPONENT_COLS + ['Rent']

# ── regression run settings ──────────────────────────────────────────────────
START_DATE        = '1990-01-01'
HORIZON           = 12                        # months ahead
MIN_TRAIN         = 120                       # 10-year minimum OOS training window
OOS_STEP          = 1                         # rolling step size (months)
N_CV_FOLDS        = 10
LAMBDA_GRID       = np.logspace(-4, 3, 40)    # comps: penalises deviation from basket prior
LAMBDA_GRID_RANKS = np.logspace(0, 7, 40)     # ranks: penalises non-smoothness across ranks
ROLLING_WINDOW    = 120                       # 10-year rolling window (optimal for Japan)

# ── matplotlib defaults ──────────────────────────────────────────────────────
MPL_RCPARAMS = {
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
}

"""
cpi_analysis.py  –  Japan CPI structure, weights, and inflation dynamics
Run:  python cpi_analysis.py
Saves figures to plots/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (
    DATA_FILE, WEIGHTS_CSV, PLOTS_DIR,
    COMPOSITE_COLS, SPECIAL_COLS, COMPONENT_COLS, ISOLATED,
    EN_TO_JPN, GROUPS, GROUP_COLORS,
)

PLOTS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})

COL_TO_GROUP = {col: grp for grp, cols in GROUPS.items() for col in cols}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['YearMonth'], format='%Y-%m')
    df = df.set_index('Date').drop('YearMonth', axis=1)
    return df.replace('-', np.nan).apply(pd.to_numeric, errors='coerce')


# ══════════════════════════════════════════════════════════════════════════════
#  BASKET WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def load_basket_weights(include_rent: bool = False) -> dict[str, float]:
    """
    Returns {english_name: basket_share} for the 46 active components.
    Correct denominator is 総合 raw weight (3,190,396,706) — not the sum
    of all rows in the CSV, which inflates the denominator with aggregates.

    Pass include_rent=True to add Rent (e.g. for full contribution decomposition).
    """
    ow = pd.read_csv(WEIGHTS_CSV)
    jpn_to_raw = dict(zip(ow['Category_Name'], ow['Weight']))
    total = jpn_to_raw['総合']

    cols = COMPONENT_COLS + (['Rent'] if include_rent else [])
    return {
        eng: jpn_to_raw[EN_TO_JPN[eng]] / total
        for eng in cols
        if EN_TO_JPN.get(eng) in jpn_to_raw
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GROWTH RATES
# ══════════════════════════════════════════════════════════════════════════════

def yoy(s: pd.Series) -> pd.Series:
    """Year-over-year % change."""
    return s.pct_change(12) * 100

def g3m3m(s: pd.Series) -> pd.Series:
    """3-month over 3-month annualized % change."""
    return ((s / s.shift(3)) - 1) * 4 * 100

def mom(s: pd.Series) -> pd.Series:
    """Month-over-month % change."""
    return s.pct_change(1) * 100


# ══════════════════════════════════════════════════════════════════════════════
#  CONTRIBUTION DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def contributions_yoy(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """
    Laspeyres contribution of each component to headline YoY (percentage points):

        contrib_i_t = w_i * (index_i_t - index_i_{t-12}) / headline_{t-12} * 100

    Summing all 47 components (including Rent) exactly reproduces headline YoY.
    Call with weights from load_basket_weights(include_rent=True).
    """
    headline = df['All items']
    return pd.DataFrame({
        col: weights[col] * (df[col] - df[col].shift(12)) / headline.shift(12) * 100
        for col in weights if col in df.columns
    })


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, bbox_inches='tight')
    print(f'  saved {name}')


def fig_core_measures(df: pd.DataFrame) -> plt.Figure:
    """
    YoY inflation: headline vs. Japan's published core measures, 1990-present.
    Shows the full history — deflation era, Abenomics, 2022 surge.
    """
    sub = df['1990':]

    series = {
        'Headline':               ('All items',                                                 '#2c3e50', '-',  2.2),
        'Core (ex fresh food)':   ('All items, less fresh food',                               '#e74c3c', '-',  1.6),
        'Core (ex food & energy)':('All items, less food (less alcoholic beverages) and energy','#3498db', '--', 1.6),
        'Core (ex imputed rent)': ('All items, less imputed rent',                             '#27ae60', ':',  1.6),
    }

    fig, ax = plt.subplots(figsize=(13, 5))
    for label, (col, color, ls, lw) in series.items():
        ax.plot(yoy(sub[col]).index, yoy(sub[col]).values,
                label=label, color=color, linestyle=ls, lw=lw)

    ax.axhline(0, color='black', lw=0.7, linestyle='--', alpha=0.4)
    ax.axhline(2, color='grey',  lw=0.8, linestyle=':', alpha=0.5)
    ax.text(pd.Timestamp('2026'), 2.1, '2% target', fontsize=7.5, color='grey')

    for ts, label in [('2013-04', 'Abenomics'), ('2022-01', '2022 surge')]:
        ax.axvline(pd.Timestamp(ts), color='grey', lw=0.8, linestyle='--', alpha=0.35)
        ax.text(pd.Timestamp(ts), ax.get_ylim()[1] * 0.85, label,
                fontsize=7.5, color='grey', rotation=90, va='top', ha='right')

    ax.set_ylabel('YoY %')
    ax.set_title('Japan CPI – Headline vs. Core Measures (YoY, 1990–present)',
                 fontsize=12, fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, 'fig1_core_measures_yoy.png')
    return fig


def fig_recent_3m3m(df: pd.DataFrame) -> plt.Figure:
    """
    3m/3m annualized: headline + cores, 2019-present.
    More responsive than YoY — shows real-time momentum during the inflation surge.
    """
    sub = df['2019':]

    series = {
        'Headline':                ('All items',                                                  '#2c3e50', '-'),
        'Core (ex fresh food)':    ('All items, less fresh food',                                '#e74c3c', '-'),
        'Core (ex food & energy)': ('All items, less food (less alcoholic beverages) and energy', '#3498db', '--'),
    }

    fig, ax = plt.subplots(figsize=(13, 4.5))
    for label, (col, color, ls) in series.items():
        ax.plot(g3m3m(sub[col]).index, g3m3m(sub[col]).values,
                label=label, color=color, linestyle=ls, lw=1.8)

    ax.axhline(0, color='black', lw=0.7, linestyle='--', alpha=0.4)
    ax.axhline(2, color='grey',  lw=0.8, linestyle=':', alpha=0.5)
    ax.set_ylabel('3m/3m annualized %')
    ax.set_title('Japan CPI – Recent Dynamics (3m/3m annualized, 2019–present)',
                 fontsize=12, fontweight='bold')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, 'fig2_recent_3m3m.png')
    return fig


def fig_basket_weights(weights: dict) -> plt.Figure:
    """
    Horizontal bar chart of all 46 active components + Rent shown separately.
    Rent is hatched and annotated to make its isolated status explicit.
    """
    # 46 active components
    w_active = pd.Series({k: v for k, v in weights.items() if k != 'Rent'}).sort_values()
    colors   = [GROUP_COLORS.get(COL_TO_GROUP.get(c, 'Other'), '#7f8c8d') for c in w_active.index]

    rent_info = ISOLATED['Rent']

    fig, ax = plt.subplots(figsize=(9, 13))

    # active components
    bars = ax.barh(range(len(w_active)), w_active.values * 100,
                   color=colors, edgecolor='white', linewidth=0.4)

    # Rent as a separate hatched bar at top
    rent_y = len(w_active) + 1
    ax.barh(rent_y, rent_info['basket_share'] * 100,
            color=GROUP_COLORS['Rent'], alpha=0.35,
            hatch='///', edgecolor=GROUP_COLORS['Rent'], linewidth=1.2)
    ax.text(rent_info['basket_share'] * 100 + 0.15, rent_y,
            f"{rent_info['basket_share']*100:.1f}%  (isolated — ~85% imputed rent)",
            va='center', fontsize=8, color=GROUP_COLORS['Rent'], style='italic')

    # value labels on bars ≥ 1%
    for i, (_, val) in enumerate(zip(bars, w_active.values)):
        if val >= 0.01:
            ax.text(val * 100 + 0.1, i, f'{val*100:.1f}%', va='center', fontsize=7.5)

    ax.set_yticks(list(range(len(w_active))) + [rent_y])
    ax.set_yticklabels(list(w_active.index) + ['Rent †'], fontsize=8)
    ax.set_xlabel('Basket share (%)')
    ax.set_title(
        'Japan CPI – Official Basket Weights (2020 base)\n'
        '46 active components sum to 81.7%.  '
        'Rent (18.3%, †) isolated due to imputed rent.',
        fontsize=11, fontweight='bold'
    )

    # group legend
    seen = {}
    for col in w_active.index:
        grp = COL_TO_GROUP.get(col, 'Other')
        if grp not in seen:
            seen[grp] = mpatches.Patch(color=GROUP_COLORS.get(grp, '#7f8c8d'), label=grp)
    seen['Rent †'] = mpatches.Patch(facecolor=GROUP_COLORS['Rent'], alpha=0.35,
                                    hatch='///', edgecolor=GROUP_COLORS['Rent'],
                                    label='Rent † (isolated)')
    ax.legend(handles=list(seen.values()), loc='lower right', fontsize=8,
              frameon=False, ncol=2)

    ax.axhline(len(w_active) + 0.5, color='grey', lw=0.8, linestyle='--', alpha=0.5)
    fig.tight_layout()
    _save(fig, 'fig3_basket_weights.png')
    return fig


def fig_contributions(df: pd.DataFrame) -> plt.Figure:
    """
    Stacked YoY contribution decomposition by group, 2018-present.
    Rent contribution drawn separately as a line so it doesn't disappear into the stack.
    Stacked components + Rent line = headline YoY exactly.
    """
    sub  = df['2018':]
    w_all = load_basket_weights(include_rent=True)
    ct   = contributions_yoy(sub, w_all).dropna(how='all')

    # group the 46 active components
    gc = {}
    for col in COMPONENT_COLS:
        grp = COL_TO_GROUP.get(col, 'Other')
        gc[grp] = gc.get(grp, pd.Series(0.0, index=ct.index)) + ct[col].fillna(0)
    gc_df = pd.DataFrame(gc)

    rent_contrib = ct['Rent'].fillna(0) if 'Rent' in ct.columns else pd.Series(0, index=ct.index)
    headline_yoy = yoy(sub['All items'])

    fig, ax = plt.subplots(figsize=(13, 5.5))

    bottom_pos = np.zeros(len(gc_df))
    bottom_neg = np.zeros(len(gc_df))
    for grp in GROUPS:
        if grp not in gc_df.columns:
            continue
        color = GROUP_COLORS[grp]
        pos = gc_df[grp].clip(lower=0).values
        neg = gc_df[grp].clip(upper=0).values
        ax.bar(gc_df.index, pos, bottom=bottom_pos, color=color, label=grp,
               width=32, align='center')
        ax.bar(gc_df.index, neg, bottom=bottom_neg, color=color,
               width=32, align='center')
        bottom_pos += pos
        bottom_neg += neg

    # Rent as a separate step line (not stacked)
    ax.plot(ct.index, rent_contrib.values,
            color=GROUP_COLORS['Rent'], lw=1.8, linestyle='--',
            label='Rent † (isolated)', zorder=6)

    ax.plot(headline_yoy.index, headline_yoy.values,
            color='black', lw=2.2, label='Headline (YoY)', zorder=7)

    ax.axhline(0, color='black', lw=0.7)
    ax.axhline(2, color='grey', lw=0.8, linestyle=':', alpha=0.5)
    ax.set_ylabel('Percentage points')
    ax.set_title(
        'Japan CPI – YoY Contribution by Group (2018–present)\n'
        'Stacked = 46 active components.  Dashed purple = Rent contribution (isolated).',
        fontsize=11, fontweight='bold'
    )
    ax.legend(loc='upper left', fontsize=8, frameon=False, ncol=4)
    fig.tight_layout()
    _save(fig, 'fig4_contributions.png')
    return fig


def fig_rent_deepdive(df: pd.DataFrame) -> plt.Figure:
    """
    Two panels:
    Top  — YoY: Rent vs. Headline vs. Core (ex fresh food).
           Shows rent barely responds even during the 2022 surge.
    Bottom — 3m/3m: Rent vs. ±1 SD band of all other 46 components.
           Shows rent is an outlier in terms of stability.
    """
    sub  = df['1990':]
    w_46 = load_basket_weights(include_rent=False)
    g_df = pd.DataFrame(
        {col: g3m3m(sub[col]) for col in w_46 if col in sub.columns}
    ).dropna(how='all')

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    # ── panel 1: YoY ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(yoy(sub['All items']).index,
            yoy(sub['All items']).values,
            color='#2c3e50', lw=2.0, label='Headline')
    ax.plot(yoy(sub['All items, less fresh food']).index,
            yoy(sub['All items, less fresh food']).values,
            color='#e74c3c', lw=1.6, linestyle='--', label='Core (ex fresh food)')
    ax.plot(yoy(sub['Rent']).index,
            yoy(sub['Rent']).values,
            color=GROUP_COLORS['Rent'], lw=2.0, label='Rent †')
    ax.axhline(0, color='black', lw=0.7, linestyle='--', alpha=0.4)
    ax.set_ylabel('YoY %')
    ax.set_title('Rent vs. Headline Inflation – YoY (1990–present)', fontweight='bold')
    ax.legend(frameon=False, fontsize=9)

    # ── panel 2: 3m/3m volatility ─────────────────────────────────────────────
    ax = axes[1]
    other_mean = g_df.mean(axis=1)
    other_std  = g_df.std(axis=1)
    rent_3m    = g3m3m(sub['Rent'])

    ax.fill_between(g_df.index, other_mean - other_std, other_mean + other_std,
                    alpha=0.12, color='#2c3e50')
    ax.plot(g_df.index, other_mean,
            color='#2c3e50', lw=1.2, linestyle='--',
            label=f'46 active components — mean ± 1 SD')
    ax.plot(rent_3m.index, rent_3m.values,
            color=GROUP_COLORS['Rent'], lw=2.2,
            label=f'Rent †  (σ = {rent_3m.dropna().std():.2f})')

    ax.axhline(0, color='black', lw=0.7, linestyle='--', alpha=0.4)
    ax.set_ylabel('3m/3m annualized %')
    ax.set_title('Rent Growth Volatility vs. All Other Components (3m/3m, 1990–present)',
                 fontweight='bold')
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    _save(fig, 'fig5_rent_deepdive.png')
    return fig


def fig_volatility_scatter(df: pd.DataFrame, weights: dict) -> plt.Figure:
    """
    Scatter: basket weight vs. 3m/3m volatility for each component.
    Rent plotted separately as a star to highlight the anomaly.
    """
    sub = df['1990':]

    rows = []
    for col, w in weights.items():
        if col not in sub.columns:
            continue
        s = g3m3m(sub[col]).dropna()
        if len(s) < 24:
            continue
        rows.append({
            'col': col, 'weight': w * 100, 'std': s.std(),
            'group': COL_TO_GROUP.get(col, 'Other'),
        })
    scatter = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))

    for grp, gdf in scatter.groupby('group'):
        ax.scatter(gdf['std'], gdf['weight'],
                   color=GROUP_COLORS.get(grp, '#7f8c8d'),
                   s=gdf['weight'] * 8 + 20, alpha=0.75,
                   edgecolors='white', lw=0.5, label=grp)

    # Rent: separate star marker
    rent_info = ISOLATED['Rent']
    ax.scatter(rent_info['sigma_3m3m'], rent_info['basket_share'] * 100,
               color=GROUP_COLORS['Rent'], marker='*', s=320, zorder=5,
               edgecolors='white', lw=0.5, label='Rent † (isolated)')
    ax.annotate('Rent †\n(18.3%, σ=1.09)',
                xy=(rent_info['sigma_3m3m'], rent_info['basket_share'] * 100),
                xytext=(2.5, 17), fontsize=8.5, color=GROUP_COLORS['Rent'],
                arrowprops=dict(arrowstyle='->', color=GROUP_COLORS['Rent'], lw=1))

    # label top-weight and highest-volatility points
    top_rows = pd.concat([scatter.nlargest(4, 'weight'),
                          scatter.nlargest(3, 'std')]).drop_duplicates()
    for _, row in top_rows.iterrows():
        ax.annotate(row['col'], xy=(row['std'], row['weight']),
                    xytext=(4, 3), textcoords='offset points', fontsize=7)

    ax.set_xlabel('Volatility – std dev of 3m/3m growth (1990–present)')
    ax.set_ylabel('Basket share (%)')
    ax.set_title(
        'CPI Components: Basket Weight vs. Volatility\n'
        'Rent ★ isolated — high weight, near-zero variance',
        fontweight='bold'
    )
    ax.legend(loc='upper right', fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    _save(fig, 'fig6_weight_vs_volatility.png')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, weights: dict) -> None:
    sub = df['1990':]
    rent_info = ISOLATED['Rent']

    print('\n' + '=' * 70)
    print('JAPAN CPI – DATA SUMMARY')
    print('=' * 70)
    print(f'  Full range:        {df.index.min():%Y-%m} – {df.index.max():%Y-%m}')
    print(f'  Regression window: 1990-01 – {df.index.max():%Y-%m}  ({len(sub)} months)')
    print(f'  Total raw columns: {len(df.columns)}')
    print(f'    Composite (excl): {len(COMPOSITE_COLS)}   — derived from headline, never use as features')
    print(f'    Special (excl):   {len(SPECIAL_COLS)}   — cross-cutting aggregates, overlap with components')
    print(f'    Active components:{len(COMPONENT_COLS):3d}   — non-overlapping, sum to 81.7%')
    print(f'    Isolated:           1   — Rent (18.3%, ~85% imputed rent)')

    print()
    print(f'  ISOLATED: Rent')
    print(f'    Total basket share:   {rent_info["basket_share"]*100:.2f}%')
    print(f'    Market rent share:    {rent_info["market_share"]*100:.2f}%  (持家の帰属家賃を除く家賃)')
    print(f'    Imputed rent share:   {rent_info["imputed_share"]*100:.2f}%  (derived, no separate series)')
    print(f'    sigma(3m/3m):         {rent_info["sigma_3m3m"]:.2f}  — lowest of all components')
    print(f'    Note: {rent_info["reason"]}')

    print()
    print('  ACTIVE COMPONENTS (46) — basket share and volatility:')
    w_series = pd.Series(weights).sort_values(ascending=False)
    for col, w in w_series.items():
        std = g3m3m(sub[col]).dropna().std() if col in sub.columns else float('nan')
        print(f'    {col:<50s}  {w*100:5.2f}%   σ={std:.2f}')

    print()
    print(f'  Active components sum: {sum(weights.values())*100:.2f}%')

    print()
    print('  RECENT HEADLINE CPI (YoY, last 12 months):')
    for date, val in yoy(df['All items']).dropna().tail(12).items():
        print(f'    {date:%Y-%m}  {val:+.2f}%')

    print()
    print('  MISSING VALUES in active components (1990–present):')
    missing = (sub[COMPONENT_COLS].isnull().sum() / len(sub) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    print('    None.' if missing.empty else
          '\n'.join(f'    {c:<50s} {p:.1f}%' for c, p in missing.items()))

    print('=' * 70 + '\n')


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    df      = load_data()
    weights = load_basket_weights(include_rent=False)

    print_summary(df, weights)

    print('Generating figures...')
    fig_core_measures(df)
    fig_recent_3m3m(df)
    fig_basket_weights(weights)
    fig_contributions(df)
    fig_rent_deepdive(df)
    fig_volatility_scatter(df, weights)

    print(f'\nDone. Figures saved to {PLOTS_DIR.resolve()}')
    plt.show()


if __name__ == '__main__':
    main()

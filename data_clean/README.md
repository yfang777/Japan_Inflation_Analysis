# data_clean/

CPI index data from the Japan Statistics Bureau (2020 base).

## Files

| File | Columns | Description |
|---|---|---|
| `level_1.csv` | 11 (1 headline + 9 sectors + 1 date) | Major CPI groups: Housing, Fuel/light/water, Furniture, Clothing, Medical, Transport/communication, Education, Culture/recreation, Miscellaneous |
| `level_2.csv` | 49 (1 headline + 48 components + 1 date) | Mid-level components (e.g. Cereals, Fish & seafood, Rent, Electricity) |
| `level_3.csv` | 745 (1 headline + 744 items + 1 date) | Granular items (e.g. Tuna, Cabbage, Gasoline). Column names need replacement from `level_3_column_names.csv` |
| `level_3_column_names.csv` | 4 | Translation table: serial, weight_2020, japanese, english. Use the `english` column as header for `level_3.csv` |

## CSV Format

All level files share the same structure:

- **Row 0 (header):** Column names (English for level 1 & 2; garbled for level 3 — use `level_3_column_names.csv`)
- **Row 1 ("Weights"):** Official 2020-base basket weights (out of 10,000). First column = `"Weights"`, rest = integer weights
- **Rows 2+:** Monthly CPI index values. First column = `YYYYMM` date string, rest = numeric index levels

The first data column after `year_month` is always `All items` (headline CPI, weight = 10,000).

## Level 3 Column Name Fix

The `level_3.csv` file has garbled column names. To fix:

1. Open `level_3_column_names.csv`
2. The `english` column contains the correct English names in order
3. Replace the header row of `level_3.csv` with these names

The mapping was translated from row 3 of `japan_cpi_level3.xlsx` (Japanese official names). Some items appear twice in the hierarchy (e.g. Milk at serial 069 and 070) — these are parent/child with the same name; disambiguate by serial number if needed.

## Weights

Weights are on a 10,000 scale (headline = 10,000). To get basket shares, divide by 10,000. For regression prior weights, divide each component's weight by the headline weight (10,000) and renormalise the non-headline components to sum to 1.

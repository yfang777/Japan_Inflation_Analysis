# level3_clean.csv — Column Removal Notes

## What was done
Removed all columns from `level_3.csv` that already appear in `level_1.csv` or `level_2.csv`.

## Summary
- **Original level_3 columns:** 745 (including year_month)
- **Columns removed:** 42
- **Columns kept:** 703 (including year_month)
- **Rows:** 676

## Removed columns (42)
These are aggregate/group-level columns that exist in level_1 or level_2:

Alcoholic beverages, All items, Bedding, Beverages, Books & other reading materials,
Cakes & candies, Cereals, Clothes, Cooked food, Culture & recreation,
Dairy products & eggs, Domestic services, Domestic utensils, Education,
Fish & seafood, Footwear, Fruits, Fuel light & water charges,
Furniture & household utensils, Household durable goods, Housing,
Interior furnishings, Meals outside the home, Meats, Medical care,
Medical services, Medical supplies & appliances, Oils fats & seasonings,
Other clothing, Other fuel & light, Personal care services, Personal effects,
Rent, Repairs & maintenance, School fees, Services related to clothing,
Shirts sweaters & underwear, Tobacco, Toilet articles, Tutorial fees,
Vegetables & seaweeds, Water & sewerage charges

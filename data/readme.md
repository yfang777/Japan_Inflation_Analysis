# Japan Inflation Data

This directory contains datasets utilized for analyzing historical inflation and Consumer Price Index (CPI) trends in Japan.

## Files Overview

### `Japan_Inflation_Structured.csv`
This structured CSV file contains comprehensive, monthly consumer price data for Japan. The time series observations start from **January 1970** (`1970-01`).

#### Data Structure & Fields
The dataset is organized chronologically by `YearMonth`, with subsequent columns representing price indices for various aggregate categories, goods, and services:

**1. Headline & Core Indices**
* `All items`
* `All items, less fresh food`
* `All items, less imputed rent`
* `All items, less fresh food and energy`

**2. Food & Beverages**
* `Food`, `Fresh food`, `Cereals`, `Fish & seafood`, `Meats`, `Dairy products & eggs`, `Vegetables & seaweeds`, `Fruits`, `Alcoholic beverages`, `Meals outside the home`.

**3. Housing & Utilities**
* `Housing`, `Rent`, `Repairs & maintenance`.
* `Fuel, light & water charges`, `Electricity`, `Gas`, `Water & sewerage charges`, `Energy`.

**4. Furniture, Clothing & Household Item**
* `Furniture & household utensils`, `Household durable goods`, `Bedding`.
* `Clothes & footwear`, `Shirts, sweaters & underwear`, `Footwear`.

**5. Medical Care**
* `Medical care`, `Medicines & health fortification`, `Medical supplies & appliances`, `Medical services`.

**6. Transportation & Communication**
* `Transportation & communication`, `Public transportation`, `Private transportation`, `Communication`.

**7. Education & Recreation**
* `Education`, `School fees`, `School textbooks & reference books for study`.
* `Culture & recreation`, `Recreational durable goods`, `Books & other reading materials`.

**8. Miscellaneous**
* `Personal care services`, `Toilet articles`, `Personal effects`, `Tobacco`.

This granular structure is designed to support in-depth analysis of which components of the Japanese economy have driven or deterred inflation over the past decades.

### `Japan_Inflation_Data.xlsx`
The original Excel workbook containing the raw inflation data, which was likely processed to generate the `Japan_Inflation_Structured.csv` file.

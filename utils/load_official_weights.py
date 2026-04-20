"""
Load Official CPI Weights from 5-1.xlsx and 2020base-list.xlsx

This script reads the official CPI weights from the Japan Statistics Bureau
and prepares them for use as prior weights in the assemblage regression model.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# File paths
DATA_DIR = Path('./data_clean')
WEIGHTS_FILE = DATA_DIR / '5-1.xlsx'
OUTPUT_FILE = DATA_DIR / 'official_weights_natianal.csv'

def load_official_weights(use_tokyo=False):
    """
    Load official CPI weights from 5-1.xlsx

    Parameters
    ----------
    use_tokyo : bool
        If True, use Tokyo area weights; otherwise use national weights

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Class_Code, Category_Name, Weight, Normalized_Weight
    """
    # Read the weights file (skip first 4 rows which are headers)
    df = pd.read_excel(WEIGHTS_FILE, header=None, skiprows=4)

    # Name the columns
    df.columns = ['Class_Code', 'Category_Name', 'Num_Items', 'National_Weight', 'Tokyo_Weight']

    # Drop rows with missing class codes
    df = df.dropna(subset=['Class_Code'])

    # Convert class code to integer
    df['Class_Code'] = df['Class_Code'].astype(int)

    # Choose which weight to use
    weight_col = 'Tokyo_Weight' if use_tokyo else 'National_Weight'
    df['Weight'] = df[weight_col]

    # Normalize weights to sum to 1
    df['Normalized_Weight'] = df['Weight'] / df['Weight'].sum()

    # Keep only relevant columns
    result = df[['Class_Code', 'Category_Name', 'Weight', 'Normalized_Weight']].copy()

    return result


def create_category_mapping():
    """
    Create a mapping between Japanese category names and English component names
    used in the inflation data.

    This mapping needs to be manually created based on the category structure.
    """
    # Complete mapping based on Japan Statistics Bureau CPI categories
    mapping = {
        # Major categories
        '総合': 'All items',
        '食料': 'Food',
        '住居': 'Housing',
        '光熱・水道': 'Fuel, light & water charges',
        '家具・家事用品': 'Furniture & household utensils',
        '被服及び履物': 'Clothing & footwear',
        '保健医療': 'Medical care',
        '交通・通信': 'Transportation & communication',
        '教育': 'Education',
        '教養娯楽': 'Culture & recreation',
        '諸雑費': 'Miscellaneous',

        # Food subcategories
        '穀類': 'Cereals',
        '魚介類': 'Fish & seafood',
        '肉類': 'Meats',
        '乳卵類': 'Dairy products & eggs',
        '野菜・海藻': 'Vegetables & seaweeds',
        '果物': 'Fruits',
        '油脂・調味料': 'Oils, fats & seasonings',
        '菓子類': 'Cakes & candies',
        '調理食品': 'Cooked food',
        '飲料': 'Beverages',
        '酒類': 'Alcoholic beverages',
        '外食': 'Meals outside the home',

        # Housing subcategories
        '家賃': 'Rent',
        '設備修繕・維持': 'Repairs & maintenance',

        # Utilities (光熱・水道)
        '電気代': 'Electric-ity',  # Note: matches data column with hyphen
        'ガス代': 'Gas',
        '他の光熱': 'Other fuel & light',
        '上下水道料': 'Water & sewerage charges',

        # Household items (家具・家事用品)
        '家庭用耐久財': 'Household durable goods',
        '室内装備品': 'Interior furnishings',
        '寝具類': 'Bedding',
        '家事雑貨': 'Domestic utensils',
        '家事用消耗品': 'Domestic non- durable goods',
        '家事サービス': 'Domestic services',

        # Clothing & footwear (被服及び履物)
        '衣料': 'Clothes',
        '和服': 'Japanese clothing',
        '洋服': 'Western clothing',
        'シャツ・セーター・下着類': 'Shirts, sweaters & underwear',
        'シャツ・セーター類': 'Shirts & sweaters',
        '下着類': 'Underwear',
        '履物類': 'Footwear',
        '他の被服': 'Other clothing',
        '被服関連サービス': 'Services related to clothing',

        # Medical care (保健医療)
        '医薬品・健康保持用摂取品': 'Medicines & health for- tification',
        '保健医療用品・器具': 'Medical supplies & appliances',
        '保健医療サービス': 'Medical services',

        # Transportation & communication (交通・通信)
        '交通': 'Public transpor- tation',
        '自動車等関係費': 'Private transpor- tation',
        '通信': 'Communi- cation',

        # Education (教育)
        '授業料等': 'School fees',
        '教科書・学習参考教材': 'School textbooks & reference books for study',
        '補習教育': 'Tutorial fees',

        # Culture & recreation (教養娯楽)
        '教養娯楽用耐久財': 'Recre- ational durable goods',
        '教養娯楽用品': 'Recre- ational goods',
        '書籍・他の印刷物': 'Books & other reading materials',
        '教養娯楽サービス': 'Recre- ational services',

        # Miscellaneous (諸雑費)
        '理美容サービス': 'Personal care services',
        '理美容用品': 'Toilet articles',
        '身の回り用品': 'Personal effects',
        'たばこ': 'Tobacco',
        '他の諸雑費': 'Other miscella- neous',

        # Composite indices
        '生鮮食品': 'Fresh food',
        '生鮮食品を除く総合': 'All items, less fresh food',
        '生鮮食品を除く食料': 'Food, less fresh food',
        '持家の帰属家賃を除く総合': 'All items, less imputed rent',
        '持家の帰属家賃及び生鮮食品を除く総合': 'All items, less imputed rent & fresh food',
        '生鮮食品及びエネルギーを除く総合': 'All items, less fresh food and energy',
        '食料（酒類を除く）及びエネルギーを除く総合': 'All items, less food (less alcoholic beverages) and energy',
        '持家の帰属家賃を除く住居': 'Housing, less imputed rent',
        '持家の帰属家賃を除く家賃': 'Rent, less imputed rent',

        # Special categories (別掲)
        '（別掲）エネルギー': 'Energy',
        '（別掲）教育関係費': 'Expenses for education',
        '（別掲）教養娯楽関係費': 'Expenses for culture & recreation',
        '（別掲）情報通信関係費': 'Expenses for information & communi-cation',

        # Special categories (うち means "of which")
        '（うち）生鮮魚介': '(of which)\nFresh fish & seafood',
        '（うち）生鮮野菜': '(of which)\nFresh vegetables',
        '（うち）生鮮果物': '(of which)\nFresh fruits',
    }

    return mapping


def match_weights_to_components(official_weights, component_names):
    """
    Match official weights to component names used in the data

    Parameters
    ----------
    official_weights : pd.DataFrame
        DataFrame with official weights
    component_names : list
        List of component names used in the inflation data

    Returns
    -------
    pd.DataFrame
        DataFrame with matched weights for each component
    """
    mapping = create_category_mapping()

    # Create reverse mapping (English to Japanese)
    eng_to_jpn = {v: k for k, v in mapping.items()}

    # Initialize results
    results = []

    for comp_name in component_names:
        # Try to find matching category
        jpn_name = eng_to_jpn.get(comp_name, None)

        if jpn_name:
            # Find weight in official data
            match = official_weights[official_weights['Category_Name'] == jpn_name]
            if not match.empty:
                weight = match['Normalized_Weight'].values[0]
                class_code = match['Class_Code'].values[0]
                matched = True
            else:
                weight = np.nan
                class_code = np.nan
                matched = False
        else:
            weight = np.nan
            class_code = np.nan
            matched = False

        results.append({
            'Component': comp_name,
            'Japanese_Name': jpn_name,
            'Class_Code': class_code,
            'Official_Weight': weight,
            'Matched': matched
        })

    return pd.DataFrame(results)


def main():
    """Main function to load and process official weights"""

    print("Loading Official CPI Weights")
    print("=" * 80)

    # Load national weights
    national_weights = load_official_weights(use_tokyo=False)
    print(f"\nLoaded {len(national_weights)} weight categories (National)")
    print("\nTop 10 categories by weight:")
    print(national_weights.nlargest(10, 'Normalized_Weight')[['Category_Name', 'Normalized_Weight']])

    # Load Tokyo weights
    tokyo_weights = load_official_weights(use_tokyo=True)
    print(f"\n\nLoaded {len(tokyo_weights)} weight categories (Tokyo)")
    print("\nTop 10 categories by weight:")
    print(tokyo_weights.nlargest(10, 'Normalized_Weight')[['Category_Name', 'Normalized_Weight']])

    # Save to CSV
    national_weights.to_csv(DATA_DIR / 'official_weights_national.csv', index=False, encoding='utf-8-sig')
    tokyo_weights.to_csv(DATA_DIR / 'official_weights_tokyo.csv', index=False, encoding='utf-8-sig')

    print(f"\n\nWeights saved to:")
    print(f"  - {DATA_DIR / 'official_weights_national.csv'}")
    print(f"  - {DATA_DIR / 'official_weights_tokyo.csv'}")

    print("\n" + "=" * 80)
    print("Next steps:")
    print("  1. Review the category mapping in create_category_mapping()")
    print("  2. Match these weights to your component names")
    print("  3. Use matched weights as w_prior in assemblage_regression()")
    print("=" * 80)


if __name__ == '__main__':
    main()

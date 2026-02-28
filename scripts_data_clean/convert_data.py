import pandas as pd
import os

print("Processing Japan Inflation Data...")
file_path = 'data/Japan_Inflation_Data.xlsx'
df = pd.read_excel(file_path, sheet_name=0, header=None)

# 1. Extract English Column Names from Row 11
english_headers_row = df.iloc[11].tolist()
japanese_headers_row = df.iloc[10].tolist()
category_headers_row = df.iloc[8].tolist()

columns = []
for i in range(len(df.columns)):
    # Column 1 contains YearMonth (e.g. 197001.0)
    if i == 1:
        columns.append('YearMonth')
    elif i < 8:
        columns.append(f'Meta_{i}')
    else:
        # Construct a meaningful header
        # prioritize English header -> Japanese header -> Category header
        header = str(english_headers_row[i]).strip() if pd.notna(english_headers_row[i]) else ''
        if not header or header == 'nan':
            header = str(japanese_headers_row[i]).strip().replace('\n', ' ') if pd.notna(japanese_headers_row[i]) else ''
        if not header or header == 'nan':
            header = str(category_headers_row[i]).strip().replace('\n', ' ') if pd.notna(category_headers_row[i]) else ''
        if not header or header == 'nan':
             header = f'Feature_{i}'
        
        # Avoid duplicate column names
        base_header = header
        counter = 1
        while header in columns:
            header = f"{base_header}_{counter}"
            counter += 1
            
        columns.append(header)

df.columns = columns

# 2. Extract Data Rows (from index 14 onwards)
data_df = df.iloc[14:].copy()

# 3. Clean Date Column
# YearMonth is currently float like 197001.0, convert to string YYYY-MM
data_df['YearMonth'] = data_df['YearMonth'].apply(lambda x: f"{int(x)//100}-{int(x)%100:02d}" if pd.notna(x) and str(x).strip() else None)

# Drop redundant "Year and month" / "年月" columns if they exist in the right side
# We only want the features based on numbers
cols_to_keep = ['YearMonth']
for col in df.columns[8:]:
    if 'Year and month' not in col and '年月' not in col:
        cols_to_keep.append(col)

data_df = data_df[cols_to_keep]

# 4. Filter out any rows without a YearMonth (like empty rows at the end)
data_df = data_df.dropna(subset=['YearMonth'])

# 5. Drop rows where 'All items' is NaN or irrelevant text
# This ensures we only get rows that have index values
data_df = data_df[pd.to_numeric(data_df['All items'], errors='coerce').notna()]

# 6. Save to CSV
os.makedirs('data_clean', exist_ok=True)
output_path = 'data_clean/japan_inflation_structured.csv'
data_df.to_csv(output_path, index=False)

print(f"Data successfully cleaned and saved to {output_path}")
print("Data shape:", data_df.shape)
print("Saved columns:", data_df.columns.tolist())
print(data_df.head())

import pandas as pd
import os

# Load the Excel file
if os.path.exists("merged_cleaned_retail_data.xlsx"):
    df = pd.read_excel("merged_cleaned_retail_data.xlsx")
    
    print("=" * 80)
    print("COLUMN NAMES AND DATA TYPES:")
    print("=" * 80)
    print(df.dtypes)
    print("\n" + "=" * 80)
    print("FIRST 5 ROWS:")
    print("=" * 80)
    print(df.head())
    print("\n" + "=" * 80)
    print("DATASET SHAPE:", df.shape)
    print("=" * 80)
else:
    print("File not found!")

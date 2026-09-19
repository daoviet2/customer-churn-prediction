from pathlib import Path

import pandas as pd


path = Path(__file__).resolve().parent / 'data' / 'raw' / 'Customer-Churn.csv'
df = pd.read_csv(path)
print('shape', df.shape)
print('columns', list(df.columns))
print(df.head().to_string(index=False))
print('\nmissing values:\n', df.isna().sum().to_string())
print('\ndtypes:\n', df.dtypes.to_string())
print('\nunique target values:', df.iloc[:, -1].dropna().unique())

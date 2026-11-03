import pandas as pd
from pathlib import Path
p = next(Path(r'c:\Users\DELL\Downloads\assignment1').glob('*.csv'))
df = pd.read_csv(p)
print(df.head())
print('\nShape:', df.shape)
print('\nColumns:', list(df.columns))
print('\nDtypes:\n', df.dtypes)
print('\nMissing values:\n', df.isnull().sum())
print('\nDescribe:\n', df.describe().T)

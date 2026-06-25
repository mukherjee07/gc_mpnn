import pandas as pd

df = pd.read_csv('new_test_set.csv')
cols_to_remove = ['He', 'H2', 'O2', 'N2', 'CO2', 'CH4']
df = df.drop(columns=cols_to_remove)
df.to_csv('new_test_set_cleaned.csv', index=False)

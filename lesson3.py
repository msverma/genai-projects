import pandas as pd

file_path = 'Sample.xlsx'  # Specify the path to your Excel file
df = pd.read_excel(file_path)
print(df.shape)
#print(df.head())
#print(df.dtypes)
missing_values = df.isnull().sum()
print("Missing values in each column:")
#print(missing_values)
msvalue= df.isnull().sum()
print(msvalue)

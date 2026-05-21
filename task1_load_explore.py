import pandas as pd
import numpy as np

# Import pandas and load churnguard_data.csv into a DataFrame
df = pd.read_csv("churnguard_data.csv")


# Print the shape of the dataset (rows, columns)
print(f'The shape of the churdguard dataset:-\n')
print(df.shape, "\n")
print("=" * 66)

# Print the first 5 rows
print(f'Printing the first 5 Rows :- \n')
print(df.head(5), "\n")
print("=" * 66)

# Print column names and data types using .info()
print(f'Checking the column names and data types:- \n')
print(df.info(), "\n" )
print("=" * 66)

# Print the count of missing values in each column
print(f'Count of missing values in each column:-\n')
print(df.isnull().sum(), "\n")
print("=" * 66)

# Print the number of duplicate rows
print(f'The total duplicated values are :- {df.duplicated().sum()}\n')
# print(f" \n")
print("=" * 66)

# Print the value counts of the Churn column — you will notice inconsistent entries
print(f"The total value counts of Churn :- \n")
print(f"{df['Churn'].value_counts()} \n")
print("=" * 66)

# Print the unique values in the Contract column — you will notice typos
print(f'The unique values in the Contract column:-\n')
print(f"{df['Contract'].unique()}")


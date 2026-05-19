import pandas as pd
# import numpy as np


df = pd.read_csv("churnguard_data.csv")


# Import pandas and load churnguard_data.csv into a DataFrame
# Print the shape of the dataset (rows, columns)
print(df.shape, "\n")

# Print the first 5 rows
print(df.head(5), "\n")

# Print column names and data types using .info()
print(df.info(), "\n" )

# Print the count of missing values in each column
print(df.isnull().sum(), "\n")

# Print the number of duplicate rows
print(f"The total duplicated values are :- {df.duplicated().sum()} \n")

# Print the value counts of the Churn column — you will notice inconsistent entries
print(f"The total value counts of Churn :- {df['Churn'].value_counts()} \n")

# Print the unique values in the Contract column — you will notice typos
print(f"{df['Contract'].unique()}")
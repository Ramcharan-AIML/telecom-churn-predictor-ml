
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Load churnguard_data.csv
df = pd.read_csv("churnguard_data.csv")

# Drop the customerID column
df = df.drop(["customerID"] , axis = 1)

# Remove duplicate rows
df = df.drop_duplicates()
# print(df.duplicated().sum())

# Strip whitespace from gender and PaymentMethod using .str.strip()
cols_del_whitespace = ["gender" , "PaymentMethod"]
df[cols_del_whitespace] = df[cols_del_whitespace].apply(lambda x: x.str.strip())
# df["PhoneService"].unique()

# Standardise casing — convert Churn, PhoneService, and PaperlessBilling to title case using .str.strip().str.title()
title_case = ['Churn' , 'PhoneService' , "PaperlessBilling"]
df[title_case] = df[title_case].apply(lambda x: x.str.strip().str.title())


# Fix Contract — map all variations to one of three valid values:
# Month-to-month, One year, Two year
contract_map = {
    "Month-to-month": "Month-to-month",
    "One year":       "One year",      
    "Two year":       "Two year",      
    "month-to-month": "Month-to-month",
    "month to month": "Month-to-month",
    "Monthly":        "Month-to-month",
    "One Year":       "One year",
    "one year":       "One year",
    "1 year":         "One year",
    "Two Year":       "Two year",
    "two year":       "Two year",
    "2 year":         "Two year",
    "2 Year":         "Two year",
}

df["Contract"] = df["Contract"].str.strip().map(contract_map)


# Fix InternetService — map all variations to one of three valid values:
# DSL, Fiber optic, No
internet_map = { "dsl":"DSL" , 
                 "fiberoptic":"Fiber optic" ,
                 "fiber optic":"Fiber optic",
                 "no":"No",
                 "fibre optic":"Fiber optic"}

df['InternetService'] = df['InternetService'].str.strip().str.lower().map(internet_map)
df['InternetService'] = df['InternetService'].fillna(df['InternetService'].mode()[0])

# Fix TotalCharges — convert to numeric using pd.to_numeric(..., errors='coerce') so junk becomes NaN
# Remove rows where tenure is zero or negative
# Remove rows where MonthlyCharges is less than 10 or greater than 200
df["TotalCharges"] = pd.to_numeric( df["TotalCharges"], errors="coerce")
df.drop(df[df['tenure'] <= 0].index, inplace=True)
df.drop(df[(df['MonthlyCharges'] < 10) | (df['MonthlyCharges'] >200)].index, inplace=True)

# Fill missing values:
# MonthlyCharges → column mean
# TotalCharges → column mean
df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].mean())
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

# tenure → column median (use integer rounding)
df['tenure'] = df['tenure'].fillna(df['tenure'].median()).astype(int)

# Print the shape of the cleaned DataFrame
print(f"The shape of the Data frame after cleaning :- {df.shape}")
print("=" * 55)

# Print missing value counts to confirm all issues are resolved
print("After completion of cleaning the data , the no.of missing values are:- ")
print(f'{df.isnull().sum()}')


# -----------------------------------------------------------------------------
# Task - 4 (Predicting by user)

# Encode Churn
df['Churn'] = df['Churn'].map({"Yes": 1, "No": 0})

# Encode Contract
contract_ordinal = {"Month-to-month": 0, "One year": 1, "Two year": 2}
df['Contract'] = df["Contract"].map(contract_ordinal)




x = df[['tenure', 'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'Contract']]
y = df['Churn']

# Scaling the full dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)          

# --- inputs ---
tenure   = int(float(input("Enter the tenure(Months):- ")))
monthly  = float(input("Enter Monthly Charges ($) :- "))
total    = float(input("Enter Total charges ($) :- "))
senior   = int(input("Senior (Yes = 1 , No = 0) :- "))
contract = int(input("Contract type (0 = Month-to-month, 1 = One year, 2 = Two year): "))

customer        = [[tenure, senior, monthly, total, contract]]
customer_scaled = scaler.transform(customer)

prediction  = model.predict(customer_scaled)[0]
probability = model.predict_proba(customer_scaled)[0]

if prediction == 1:
    print("  Prediction: This customer is likely to CHURN.")
    print(f"  Churn probability : {probability[1]:.1%}")
    print(f"  Stay  probability : {probability[0]:.1%}")
else:
    print("  Prediction: This customer is likely to STAY.")
    print(f"  Stay  probability : {probability[0]:.1%}")
    print(f"  Churn probability : {probability[1]:.1%}")


import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score , recall_score, precision_score, f1_score
from sklearn.metrics import classification_report


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


# -------------------------------------------------------------------------------


# Task 3 — Train a Classification Model


# Encode the target column Churn:
# Yes → 1, No → 0
label_enc = LabelEncoder()
df["Churn"] = label_enc.fit_transform(df['Churn'])

# Encode categorical columns using pd.get_dummies() with drop_first=True:
# gender, PhoneService, InternetService, Contract, PaperlessBilling, PaymentMethod
cat_cols = ['gender','PhoneService', 'InternetService',
       'Contract', 'PaperlessBilling', 'PaymentMethod']
df = pd.get_dummies(df , columns=cat_cols , drop_first=True)


# Separate the data into:
# X — all columns except Churn
# y — the Churn column
x = df.drop(['Churn'] , axis= 1)
y = df['Churn']

# Split into train and test sets — 80% train, 20% test, random_state=42
X_train , X_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

# Train a LogisticRegression model with max_iter=1000
model = LogisticRegression(max_iter=1000)
model.fit(X_train , y_train)

y_pred = model.predict(X_test)
y_pred

# Print the accuracy score on the test set
acc = accuracy_score(y_test , y_pred)

print("ACCURACY SCORE :-")
print(f"The accuracy of the test and prediction is :- {acc:.4f}")

# Print the classification report using classification_report with target_names=['Stay', 'Churn']
report = classification_report(y_test , y_pred , target_names=['Stay' , "Churn"])
print("The classification report for the data :- ")
print(f"{report}")


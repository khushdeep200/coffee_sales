# COFFIE_SALES DATA ANALYSIS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

#import dataset
data=pd.read_csv("index.csv")

#basic EDA
print("\n First 10 records of dataset")
print(data.head(10))
print("\n Last 10 records of dataset")
print(data.tail(10))
print("\n Statistic summary ")
print(data.describe())
print("\n summary information")
print(data.info())
print("\n Number of rows and columns(shape)")
print(data.shape)

#check for missing values
print("\nCheck for missing values")
print(data.isnull().sum())

#converting datatypes
data['datetime']=pd.to_datetime(data['datetime'])

#the only column that have null values is card, which reflects cash payment.

#outlier check
df=data[(abs(zscore(data['money']))>3)]
print ("\n Outliers:",df)

#feature engineering
#extract month, year, time from the datetime
data['month']=data['datetime'].dt.month
data['year']=data['datetime'].dt.year
data['hour']=data['datetime'].dt.hour


#1st objective: most busy hours of days
fig, ax= plt.subplots(figsize=(10,6))
fig.patch.set_facecolor("#dbe9f4")
ax.set_facecolor('#ffffff')
sns.set(style="whitegrid")
sns.histplot( data=data, x="hour", bins=24, discrete=True, color="#1f77b4",
              edgecolor="black", ax=ax)
ax.set_xticks(range(0,24))
plt.title("Most busy hours of the day")
plt.tight_layout()
plt.show()


#2nd objective: sales per month
import calendar
monthly_avg = data.groupby('month')['money'].sum().reset_index()
monthly_avg['month'] = monthly_avg['month'].apply(lambda
                                                  x: calendar.month_abbr[x])
month_order = list(calendar.month_abbr)[1:]  # ['Jan', 'Feb', ... 'Dec']
monthly_avg['month'] = pd.Categorical(monthly_avg['month'],
                                      categories=month_order, ordered=True)
monthly_avg = monthly_avg.sort_values('month')
fig, ax= plt.subplots(figsize=(10,6))
fig.patch.set_facecolor("#dbe9f4")
ax.set_facecolor('#ffffff')
sns.set(style="whitegrid")
sns.lineplot(data=monthly_avg, x='month', y='money', marker='o',
             color="#1f77b4")
ax.set_title("Total Sales Per Month", fontsize=16, weight="bold")
ax.set_xlabel("Month", fontsize=14)
ax.set_ylabel("Total Sales (Money)", fontsize=14)
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

#3rd objective
g = data['coffee_name'].value_counts().reset_index()
g.columns = ['coffee_name', 'count']
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#dbe9f4')
ax.set_facecolor('#ffffff')
sns.set(style="whitegrid")
plt.pie(
    g['count'], 
    labels=g['coffee_name'], 
    autopct='%1.1f%%', 
    startangle=140
)
plt.title("Coffee Sales Distribution by Type", fontsize=14)
plt.tight_layout()
plt.show()

#4th objective
sales_by_type = data.groupby("coffee_name")["money"].sum().reset_index().sort_values(by="money", ascending=False)
total_revenue = data["money"].sum()
fig, ax= plt.subplots(figsize=(10,6))
fig.patch.set_facecolor("#dbe9f4")
ax.set_facecolor('#ffffff')
sns.set(style="whitegrid")
sns.barplot(x="coffee_name",y="money",data=sales_by_type, hue="coffee_name", legend=False,palette="copper_r")
plt.title("Total Sales by Coffee Type", fontsize=14)
plt.xticks(rotation=45)
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()

#5th objective
favorite_coffee = data.groupby(["card", "coffee_name"])["money"].sum().reset_index()
favorite_coffee = favorite_coffee.sort_values(by=["card", "money"], ascending=[True, False])
favorite_per_customer = favorite_coffee.groupby("card").head(1)
top_customers = favorite_per_customer.sort_values("money", ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x="money", y="card", hue="coffee_name", data=top_customers, dodge=False, palette="Set2" )
plt.title("Top 10 Customers and Their Favorite Coffee", fontsize=14)
plt.xlabel("Total Spending on Favorite Coffee")
plt.ylabel("Customer (Card ID)")
plt.legend(title="Favorite Coffee")
plt.tight_layout()
plt.show()

#6th objective
payment_counts = data["cash_type"].value_counts()
print("\n🧾 Purchases by Payment Method:\n", payment_counts)

plt.figure(figsize=(6,6))
plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Purchases by Payment Method")
plt.show()

# 7th objective: Weekly and Daily Sales Patterns

# Extract weekday (0=Monday, 6=Sunday)

data['weekday'] = data['datetime'].dt.day_name()

# --- Weekly Sales Pattern ---
weekly_sales = data.groupby("weekday")["money"].sum().reset_index()

# To order weekdays properly
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
weekly_sales['weekday'] = pd.Categorical(weekly_sales['weekday'],
                                         categories=weekday_order, ordered=True)
weekly_sales = weekly_sales.sort_values("weekday")

plt.figure(figsize=(10,6))
sns.barplot(x="weekday", y="money", data=weekly_sales,hue="weekday",
            palette="viridis_r")
plt.title("Weekly Sales Pattern", fontsize=14, weight="bold")
plt.xlabel("Day of the Week")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#8th objective sales prediction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# copy dataset
df_ml = data.copy()

# Drop datetime column (can’t be directly used in regression)
if "datetime" in df_ml.columns:
    df_ml = df_ml.drop(columns=["datetime"])

# Encode categorical columns
le = LabelEncoder()
for col in ["coffee_name", "cash_type", "weekday"]:
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

# Define features (X) and target (y)
X = df_ml[["hour", "month", "year", "coffee_name", "cash_type", "weekday"]]
y = df_ml["money"]

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully")
print("R^2 Score:", model.score(X_test, y_test))

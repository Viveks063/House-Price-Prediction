import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import streamlit as st
import joblib

dataset = pd.read_csv('Housing.csv')
print(dataset.head())

print(dataset.shape)

print(dataset.isnull().sum())

print(dataset.describe())

numeric_ds = dataset.select_dtypes(include=[np.number])

correlation = numeric_ds.corr()

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
plt.show()

X = dataset.drop(['price'], axis = 1)
Y = dataset['price']
print(X)
print(Y)
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

model = XGBRegressor()

model.fit(X_train, Y_train)

house_price_prediction_train = model.predict(X_train)

print(house_price_prediction_train)

score_1 = metrics.r2_score(Y_train, house_price_prediction_train)

score_2 = metrics.mean_absolute_error(Y_train, house_price_prediction_train)

print("R squared error : ", score_1)
print("Mean Absolute error : ", score_2)

house_price_prediction_test = model.predict(X_test)

print(house_price_prediction_train)

score_1 = metrics.r2_score(Y_test, house_price_prediction_test)

score_2 = metrics.mean_absolute_error(Y_test, house_price_prediction_test)

print("R squared error : ", score_1)
print("Mean Absolute error : ", score_2)

plt.scatter(Y_train, house_price_prediction_train)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted")
plt.show()

joblib.dump(model, "house_price_prediction.pkl")

model = joblib.load("house_price_prediction.pkl")

st.title("House Price Prediction")

st.sidebar.header("Input Features")

def user_inputs():
    bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
    bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
    area = st.sidebar.number_input("Area in sq ft", min_value=500, max_value=10000, step=100)

    input_data = {
        "bedrooms" : bedrooms,
        "bathrooms" : bathrooms,
        "area" : area
    }

    return pd.DataFrame(input_data, index=[0])

input_data = user_inputs()

encoded_data = pd.get_dummies(input_data, drop_first=True)

for col in X.columns:
    if col not in encoded_data.columns:
        encoded_data[col] = 0
encoded_data = encoded_data[X.columns]

if st.button("Predict"):
    prediction = model.predict(encoded_data)
    st.write(f"### Predicted House Price: ${prediction[0]:,.2f}")

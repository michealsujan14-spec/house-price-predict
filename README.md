# house-price-predict
this ml model is to predict the price of the house in various places


# ==========================================
# HOUSE PRICE PREDICTION - PRODUCTION VERSION
# ==========================================
from re import X

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("house_price.csv")


x = df.drop("Price" , axis = 1)
y = df["Price"]

# Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2 )

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print(predictions)
import datetime
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics

data = pd.read_csv('SuperMarketSales.csv')
print(data.describe())
model = linear_model.LinearRegression()
cols = ["Store", "Temperature", "Fuel_Price", "CPI"]
mine = 999999999999999999999999
best = ""

for s in range(0, 4):
    Inp = data[f'{cols[s]}']
    Out = data['Weekly_Sales']
    Inp = np.expand_dims(Inp, axis=1)
    Out = np.expand_dims(Out, axis=1)
    model.fit(Inp, Out)
    prediction = model.predict(Inp)
    Mse = metrics.mean_squared_error(Out, prediction)
    print(f'Mean Square Error for {cols[s]} = ', Mse)
    if mine > Mse:
        mine = Mse
        best = f"{cols[s]}"

data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data['Date_Year'] = data['Date'].dt.year

Inp = data['Date_Year']
Out = data['Weekly_Sales']

Inp = np.expand_dims(Inp, axis=1)
Out = np.expand_dims(Out, axis=1)
model.fit(Inp, Out)
prediction = model.predict(Inp)
Mse = metrics.mean_squared_error(Out, prediction)
print(f'Mean Square Error for Date = ', Mse)

if mine > Mse:
    mine = Mse
    best = f"{cols[s]}"
print(f"The minimum Mean Squared error is {best} and = {mine}")

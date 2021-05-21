import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

#LOADING DATA
data = pd.read_csv('new data.csv',sep = ',')
data = data[['Day','Confirmed']]
print(data.head())   #printing the first five data

#PREPARING DATA

x = np.array(data['Day']).reshape(-1, 1)
y = np.array(data['Confirmed']).reshape(-1, 1)
polyFeature = PolynomialFeatures(degree=5)
x = polyFeature.fit_transform(x)

#TRAINING DATA
print('TRAINING DATA\n')
model = linear_model.LinearRegression()
model.fit(x,y)
accuracy = model.score(x,y)
print(f'Accuracy:{round(accuracy*100,2)} %')
y0 = model.predict(x)
#PREDICTION
days = 7
print('-'*30);print('PREDICTION');print('-'*30)
print(f'Prediction of Cases in Rajasthan  after {days} days:',end='')
print(round(int(model.predict(polyFeature.fit_transform([[443+days]])))/100000,2),'+lakhs')




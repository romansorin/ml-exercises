import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.neighbors

# Simple univariate linear regression: attempt to predict house price based on age (X = age, Y = price)
x_axis_label = "X2 house age" # Choose any column label as a variable; from testing, house age on its own isn't a very good predictor
y_axis_label = "Y house price of unit area"

re_data = pd.read_csv("./data/real_estate.csv")
X = np.c_[re_data[x_axis_label]]
y = np.c_[re_data[y_axis_label]]

re_data.plot(kind='scatter', x=x_axis_label, y=y_axis_label)
plt.show()

model = sklearn.linear_model.LinearRegression()

model.fit(X, y)

X_new = [[15]]
y_new = model.predict(X_new)
print(y_new)

model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
model.fit(X,y)
y_new = model.predict(X_new)
print(y_new)

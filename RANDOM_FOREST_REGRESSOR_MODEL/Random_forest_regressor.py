
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

Data = pd.read_csv(r"C:\Users\Dell\Downloads\emp_sal.csv")

x = Data.iloc[:,1:2].values
y = Data.iloc[:,2].values 

rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(x,y)

plt.scatter(x,y, color = "red")
plt.plot(x, rf_model.predict(x), color="green")
plt.title("Rnadom Forest Regressor")
plt.xlabel(" position level")
plt.ylabel("salary")
plt.show()

rf_model = rf_model.predict([[6.5]])
rf_model


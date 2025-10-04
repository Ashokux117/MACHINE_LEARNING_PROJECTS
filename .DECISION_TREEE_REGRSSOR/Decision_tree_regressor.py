import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


Data = pd.read_csv(r"C:\Users\Dell\Downloads\emp_sal.csv")

x = Data.iloc[:,1:2].values
y = Data.iloc[:,2].values

dt_model =  DecisionTreeRegressor()
dt_model.fit(x,y)

dt_model_pred = dt_model.predict([[6.5]])
print(dt_model_pred)

plt.scatter(x, y, color="red")
plt.plot(x, dt_model.predict(x),color = "blue")
plt.title('Decision Tree Regression')
plt.xlabel("position label")
plt.ylabel("salary")
plt.show()


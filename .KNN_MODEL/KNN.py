
#
# import numerical libraries
import pandas as pd
import numpy as np 
 
# import plotting libraries
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsRegressor 

# Load dataset
data = pd.read_csv(r'C:\Users\DELL\Downloads\emp_sal.csv')
x = data.iloc[:,1:2].values
y = data.iloc[:,2].values

# Train KNN model
knn_model = KNeighborsRegressor()   # you can change k
knn_model.fit(x, y)

# Make prediction
knn_pred = knn_model.predict([[6.5]])
knn_pred

# Visualization
x_grid = np.arange(min(x), max(x), 0.1).reshape(-1, 1)

plt.scatter(x, y, color="red")
plt.plot(x_grid, knn_model.predict(x_grid), color="blue")
plt.title("KNN Regression Model")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

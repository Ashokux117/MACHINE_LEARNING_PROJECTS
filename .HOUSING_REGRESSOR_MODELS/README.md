# House Price Prediction Web App

This is a Flask-based machine learning web application that predicts house prices using multiple regression models.
It allows users to input property details and select from several trained ML models to get predictions.

## 🚀 Features

- Supports multiple models (Linear, Ridge, Lasso, ElasticNet, RandomForest, XGBoost, etc.)
- User-friendly web interface built with Flask templates (`index.html`, `results.html`, `model.html`)
- Displays model evaluation results from `model_evaluation_results.csv`
- Uses pre-trained models stored as `.pkl` files

## 🧠 Models Used

- LinearRegression
- RobustRegression
- RidgeRegression
- LassoRegression
- ElasticNet
- PolynomialRegression
- SGDRegressor
- ANN
- RandomForest
- SVM
- LGBM
- XGBoost
- KNN

## 🗂️ Project Structure


├── app.py

├── model_evaluation_results.csv

├── templates/

│   ├── index.html

│   ├── results.html

│   └── model.html

├── static/

│   └── (optional CSS, JS files)

├── *.pkl (model files)

├── requirements.txt

└── .github/workflows/deploy.yml


## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Ashokux17/houseprice-prediction.git
   cd house-price-prediction
   ```
2. Create a virtual environment and install dependencies:
   pip install -r requirements.txt
3. Run the Flask app:
   ` python App.py`
4. Open  browser and visit:
   http://127.0.0.1:5000/

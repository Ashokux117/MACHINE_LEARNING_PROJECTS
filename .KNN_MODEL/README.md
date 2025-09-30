# KNN Regression Visualizer

An interactive **K-Nearest Neighbors (KNN) Regression** web application built with [Streamlit](https://streamlit.io/).  
This app allows users to upload a dataset, choose variables, visualize the regression fit, and make predictions.

##  Features
- Upload your own CSV file.
- Select independent (`X`) and dependent (`Y`) variables.
- Adjust the number of neighbors `k` with a slider.
- Visualize the regression line vs. raw data points.
- Make predictions for custom input values.

## Installation & Setup

## 1. Clone this repository
```bash
git clone https://github.com/Ashokux117/MACHINE_LEARNING_PROJECTS.git

## create a vertual environment
venv\Scripts\activate

##Install dependencies
pip install -r requirements.txt

## Run The App
streamlit run app.py

## project structure
MACHINE_LEARNING_PROJECTS/
└── KNN_MODEL/
    ├── knn.py                # Trained Knearest neighbour Regression model
    ├── README.md             # Project Documentation
    ├── ci.yml                #
    ├── emp_sal_clean.csv     # Dataset used for training
    ├── knn_app.py            # streamlit app file
    └── requirements.txt      # python dependencies



# Salary Prediction App

This is a **Streamlit-based web application** that predicts salary based on years of experience using a **Linear Regression model**.

## Features
- Simple and intuitive user interface with Streamlit.
- Takes **years of experience** as input and predicts salary.
- Trained using a sample dataset (`Salary_Data.csv`) and saved as a **pickle file** (`lrg_model.pkl`).

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Ashokux117/MACHINE_LEARNING_PROJECTS.git

2. Install dependencies:
pip install -r requirements.txt


 3. Make sure lrg_model.pkl is present in the same folder as app.py.
Run the app
streamlit run app.py

4. project structure
MACHINE_LEARNING_PROJECTS/
└── linear_regression/
    ├── Linear_reg_model.py
    ├── Liear_reg_model_part_2.py
    ├── README.md                  # Project documentation
    ├── Salary_Data.csv            # dataset 
    ├── app.yml                    # streamlit app run
    ├── lrg.py                     
    ├── lrg_model.pkl              # backend pkl file
    ├── requirements.txt             


5. Dependencies

Python 3.10

Streamlit

NumPy

scikit-learn



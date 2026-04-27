# Student Performance Prediction App

## Overview

This project predicts student performance using machine learning techniques based on behavioral and academic data.

The goal is to identify students at risk and support data-driven decision making in educational environments.

---

## Objective

* Predict student success based on input features
* Identify students at risk of poor performance
* Provide interpretable results for better understanding

---

## Features Used

- log_active_days  
- mean_score  
- log_total_clicks  
- studied_credits  
- num_of_prev_attempts  
- highest_education  
- imd_band  
- age_band  
- disability  

---

## Model

* Random Forest Classifier (best performing model)
* Compared with:

  * Decision Tree
  * Logistic Regression

Random Forest was selected due to its strong performance and ability to handle complex patterns.

---

## Output

The application provides:

* Predicted class (Pass / Fail or Risk Level)
* Probability score
* Risk level categorization (Low / Medium / High)

---

## Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Joblib

---

## Installation

Install required libraries:

- Python 3.8+
- streamlit
- scikit-learn
- pandas
- numpy
- joblib

---

## How to Run

1. Open a terminal and navigate to the project folder:

```bash
cd path/to/project
```

2. Run the Streamlit application:

```bash
streamlit run student_prediction_app.py
```

3. Access the app:

* A local server will start automatically
* Open the link shown in the terminal (usually http://localhost:8501)

---

## Key Highlights

* End-to-end pipeline: data preprocessing → model training → deployment
* Handles real-world data challenges such as missing values and feature encoding
* Interactive UI built using Streamlit


---

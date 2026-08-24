# Loan Approval Prediction

A Flask web application that uses a **Support Vector Classifier (SVC)** to predict loan approval outcomes.

## Features

* Machine learning loan approval prediction
* SVC model with probability estimates
* Data preprocessing and validation
* Model performance metrics
* CSV dataset upload and model retraining
* REST API with Swagger documentation
* Web dashboard and logging

## Tech Stack

**Python · Flask · scikit-learn · Pandas · NumPy · Matplotlib · Flasgger**

## Run Locally

```bash
git clone https://github.com/alexhp84/Loan-Site.git
cd Loan-Site
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000**

Swagger API documentation:

**http://localhost:5000/apidocs/**

## Model

The SVC model uses applicant income, loan amount, credit history, credit score, home ownership and previous loan defaults to predict approval.

## Author

**Alex Patnick**

> Educational and portfolio project. Predictions should not be used for real-world lending decisions.

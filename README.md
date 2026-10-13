# Loan Approval Prediction

## Overview
A machine learning project that predicts loan approval status based on applicant information. It includes data analysis, model training, evaluation, and deployment through a FastAPI REST API.

## Tech Stack
- Python
- Pandas, Matplotlib, Seaborn
- Scikit-learn
- FastAPI, Pydantic, Uvicorn
- Joblib, Jupyter Notebook

## Workflow
1. Data cleaning and exploratory data analysis.
2. Feature encoding and scaling.
3. Training and tuning Logistic Regression, KNN, and SVM classifiers.
4. Building and saving a preprocessing and prediction pipeline.
5. Deploying the model using FastAPI.

## Model Performance

Evaluation on 62 test samples:

| Metric | Score |
|---|---:|
| Accuracy | 80.65% |
| Weighted Precision | 81.08% |
| Weighted Recall | 80.65% |
| Weighted F1-score | 79.07% |

## API Usage

Start the API:

```bash
uvicorn main:app --reload
```

- Endpoint: `POST /predict/`
- API documentation: `http://127.0.0.1:8000/docs`

Example request:

```json
{
  "Married": "Yes",
  "Education": "Graduate",
  "ApplicantIncome": 5000,
  "LoanAmount": 150,
  "CreditHistory": 1
}
```

## Project Structure

```text
├── data/loan_data.csv
├── output/pipeline.pkl
├── main.py
├── requirements.txt
└── README.md
```

## Disclaimer
This project is for educational purposes and is not intended for making real-world lending decisions.

## Data

[Kaggle Dataset](https://www.kaggle.com/datasets/bhavikjikadara/loan-status-prediction)
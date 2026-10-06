from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

class InputData(BaseModel):
    Married: str
    Education: str
    ApplicantIncome: float
    LoanAmount: float
    CreditHistory: float

model = joblib.load('output/pipeline.pkl')

app = FastAPI()

@app.post('/predict/')
def predict(input_data: InputData):
    
    input_df = pd.DataFrame([{
        "Married": input_data.Married,
        "Education": input_data.Education,
        "ApplicantIncome": input_data.ApplicantIncome,
        "LoanAmount": input_data.LoanAmount,
        "CreditHistory": input_data.CreditHistory,
    }])
    
    pred = int(model.predict(input_df)[0])
    
    return {'prediction': pred}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
from pydantic import BaseModel
from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('diabetes_scaler.pkl')

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post('/predict')
def predict(data: PatientData):
    input_data = [[
        data.Pregnancies, data.Glucose, data.BloodPressure,
        data.SkinThickness, data.Insulin, data.BMI,
        data.DiabetesPedigreeFunction, data.Age
    ]]
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    return {"prediction": int(prediction[0])}

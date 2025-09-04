import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI(title="Iris Classifier API")

model = None
model_path = os.path.join("model", "iris_model.pkl")
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")

@app.get("/")
def home():
    return {"message": "Iris Classifier API. Use /predict to make a prediction."}

@app.post("/predict")
def predict(data: IrisData):
    if model is None:
        return {"error": "Model not loaded."}
    input_data = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prediction = model.predict(input_data)[0]
    return {"prediction": int(prediction)}

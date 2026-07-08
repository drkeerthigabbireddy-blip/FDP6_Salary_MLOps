from fastapi import FastAPI
from pydantic import BaseModel

import mlflow
import pandas as pd

# import nest_asyncio
import uvicorn
import nest_asyncio
import logging


#Configure logging
logging.basicConfig(filename = "salary1_api.log",level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",force=True)

#Load ML FLow Model
MODEL_URI = "models:/Salary Prediction model/3"
print("Loading Registered Model...")
logging.info("Loading Registered Model...")
model = mlflow.pyfunc.load_model(MODEL_URI)
print("Model Loaded Successfully!")
logging.info("Model Loaded Successfully!")

## create fast api

app =FastAPI (title = "Salary Prediction API",
              version = "1.0",
              description = "Salary Prediction using MLflow Registered Model")
#Schema

class Employee(BaseModel):
    Age: int
    Gender: str
    Education_Level: str 
    Job_Title: str 
    Years_of_Experience: float

@app.get("/health")
def health():
    logging.info("health endpoint accessed.")
    return {
        "status" : "Running",
        "model" : "Salary Prediction Version 3"
    }


@app.post("/predict")
def predict(employee: Employee):
    # Create a DataFrame from the incoming employee data
    try:
        logging.info("Prediction Request Received")
        logging.info(employee.dict())
        input_df = pd.DataFrame([{
            "Age": employee.Age,
            "Gender": employee.Gender,
            "Education Level": employee.Education_Level,
            "Job Title": employee.Job_Title,
            "Years of Experience": employee.Years_of_Experience
            }])

        # Generate prediction using the loaded model
        prediction = model.predict(input_df)
        logging.info(f"Prediction:{prediction[0]}")
        # Return the result rounded to 2 decimal places
        return {
        "Predicted Salary": round(float(prediction[0]), 2)
        }
    except Exception as e:
        logging.error(str(e))
        return{"Error":str(e)}


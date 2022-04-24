import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import os

# Create an app
app = FastAPI()
# Read a model
file_name = os.path.join(os.path.dirname(__file__), "model.joblib")
rf = joblib.load(file_name)


class Item(BaseModel):
    sepal_length: int
    sepal_width: int
    petal_length: int
    petal_width: int



@app.post("/predict")
async def predict(item:Item):
    data = [[item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]]
    prediction = rf.predict(data)
    return {"message": f"{prediction}"}
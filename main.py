
import json
from plistlib import load
from typing import Optional
from DataModel import DataModelList, DataModel
from typing import FrozenSet
import pandas as pd
import numpy as np
from pandas import json_normalize
from sklearn.metrics import r2_score, mean_squared_error
from joblib import load
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI
from sklearn.metrics import confusion_matrix, accuracy_score

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Realizar predicciones con el modelo
@app.post("/predict")
def make_predictions(data: DataModelList):
    an = jsonable_encoder(data)
    df = json_normalize(an['data'])
    df.columns = DataModel.columns()
    model = load("assets/pipeline2.joblib")
    result = model.predict(df['study_and_condition'])
    lists = result.tolist()
    json_predict = json.dumps(lists)
    return {"Predict": json_predict}



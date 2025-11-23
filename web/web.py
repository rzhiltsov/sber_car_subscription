import pandas as pd
from cloudpickle import cloudpickle
from fastapi import FastAPI, Response

from response import Metadata, Form, Prediction

app = FastAPI()

with open('data/pipeline.pkl', 'rb') as file:
    pipe: dict = cloudpickle.load(file)


@app.get('/health')
def health():
    return Response()


@app.get('/version', response_model=Metadata)
def version():
    return pipe['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    x = pd.json_normalize(form.model_dump())
    y = pipe['pipeline'].predict(x)

    return Prediction(target_action=y[0])

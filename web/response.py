from datetime import datetime

from pydantic import BaseModel


class Metadata(BaseModel):
    name: str
    author: str
    version: int
    date: datetime
    type: str
    accuracy: float
    roc_auc: float


class Form(BaseModel):
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    target_action: int

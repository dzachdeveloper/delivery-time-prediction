"""FastAPI service for Food Delivery Time Prediction."""
import logging
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

FEATURES = ['distance_km', 'prep_time_min', 'num_items', 'hour', 'is_raining', 'rider_experience_months']

# TODO 3d: BUFFER_MIN is added to the prediction before we show the time to the customer.
#   Option A: "0"  -> show the raw prediction: about half of orders will arrive late
#   Option B: "5"  -> a safety margin, most orders arrive on or before the promise
#   Option C: "10" -> very safe, but the app looks slow next to competitors
# THINK: why is the best promise usually NOT the model's best guess?
DEFAULT_BUFFER = "5"
if DEFAULT_BUFFER == "7":
    raise RuntimeError("TODO 3d in app.py: set DEFAULT_BUFFER (minutes added to each promise)")
BUFFER_MIN = float(os.getenv("BUFFER_MIN", DEFAULT_BUFFER))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("delivery-eta-api")


def load_model():
    uri = os.getenv("MODEL_URI")                 # e.g. models:/delivery-eta-model@production
    if uri:
        import mlflow.sklearn
        return mlflow.sklearn.load_model(uri), uri
    path = os.getenv("MODEL_PATH", "model.joblib")
    return joblib.load(path), path


model, MODEL_SOURCE = load_model()
log.info("Model loaded from %s", MODEL_SOURCE)

app = FastAPI(title="Food Delivery Time Prediction API", version="1.0.0")


# TODO 3a: set limits (ge = minimum, le = maximum) so impossible values get a 422 error.
# One field is done for you as an example. Training data ranges (from df.describe() in Part 1):
#   distance_km                0.3 - 20
#   prep_time_min              4 - 31
#   num_items                  1 - 11
#   hour                       10 - 23
#   is_raining                 0 - 1
#   rider_experience_months    0 - 96
# Options for how wide to make the limits:
#   Option A: exactly the training min/max  -> safest for the model, but rejects some real requests
#   Option B: real-world limits             -> e.g. 0-1 for yes/no fields, 0-23 for hours
# THINK: what should happen to a value that is possible but outside the training data? Reject? Warn?
class Request(BaseModel):
    distance_km: float = Field(..., ge=0.1, le=50, description="Distance from restaurant to customer (km)", examples=[9.0])
    prep_time_min: float = Field(..., ge=1, le=120, description="Kitchen preparation time (minutes)", examples=[20.0])
    num_items: int = Field(..., ge=1, le=50, description="Items in the order", examples=[6])
    hour: int = Field(..., ge=0, le=23, description="Hour the order was placed (0-23)", examples=[19])
    is_raining: int = Field(..., ge=0, le=1, description="1 = raining", examples=[1])
    rider_experience_months: float = Field(..., ge=0, le=480, description="Rider experience (months)", examples=[2.0])


class Prediction(BaseModel):
    predicted_minutes: float
    promised_minutes: int
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    # TODO 3c: return a dict with the keys "model_source", "buffer_minutes" and "features".
    # HINT: the values are MODEL_SOURCE, BUFFER_MIN and FEATURES (all defined above).
    # THINK: who would call this endpoint, and when? (support staff checking a late-order complaint?)
    return {"model_source": MODEL_SOURCE, "buffer_minutes": BUFFER_MIN, "features": FEATURES}
    

@app.post("/predict", response_model=Prediction)
def predict(request: Request):
    start = time.perf_counter()
    X = pd.DataFrame([request.model_dump()])[FEATURES]
    minutes = float(model.predict(X)[0])
    promised = int(round(minutes + BUFFER_MIN))

    # TODO 3b: write the message the customer sees in the app.
    #   Option A (single time):  f"Arriving in about {promised} minutes"
    #   Option B (time window):  f"Arriving in {int(minutes)}-{promised} minutes"
    #   Option C (your own):     e.g. add "(busy time, thanks for waiting)" when the order is placed at rush hour
    # THINK: does a window feel more honest to customers, or more vague?
    message = f"Arriving in {int(minutes)}-{promised} minutes"

    log.info("predict minutes=%.1f promised=%d latency_ms=%.1f", minutes, promised, (time.perf_counter() - start) * 1000)
    return Prediction(predicted_minutes=round(minutes, 1), promised_minutes=promised, message=message)

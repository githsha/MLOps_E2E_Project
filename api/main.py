from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import os
import boto3

# ---------------------------
# CHANGE THESE IF NEEDED
# ---------------------------
S3_BUCKET="aws-s3-bucket-dec-15"
S3_KEY="files/md5/6a/ce25483ff0f3998d94aa67f1635673"
LOCAL_MODEL_PATH="models/model.pkl"
# ---------------------------

app = FastAPI()

def download_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, S3_KEY, LOCAL_MODEL_PATH)

# Download model ONCE at startup
download_model()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

model, vectorizer = joblib.load("models/model.pkl")

@app.post("/predict")
def predict(request: PredictRequest):
    vec = vectorizer.transform([request.text])
    prediction = model.predict(vec)[0]
    return {"prediction": prediction}
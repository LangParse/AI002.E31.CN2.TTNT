import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from contextlib import asynccontextmanager


# Định nghĩa class cho input data (dựa trên features) - đã thêm các features mới từ preprocessing
class PredictionInput(BaseModel):
    age: int
    gender: int
    occupation: str
    income_level: str
    living_status: str
    years_with_hypertension: int
    comorbidities: str
    medication_name: str
    dose_mg: int
    salty_food_often: int
    smoking_status: int
    alcohol_habit: int
    caffeine_habit: int
    late_sleep_habit: int
    stress_level: int
    late_night: int
    sleep_hours: float
    sodium_intake_g: float
    exercise_minutes: int
    caffeine_mg: int
    alcohol_ml: int
    scheduled_time_min: float
    time_taken_min: float
    bed_time_min: float
    time_deviation: float
    time_before_bed: float
    is_morning_med: int
    is_evening_med: int
    bp_systolic: int
    bp_diastolic: int
    bp_pulse_pressure: int
    bp_mean_arterial: float
    bp_category: str
    bp_abnormal: int
    age_group: str
    is_elderly: int


def save_model(pipeline, model_path="../models/best_model.pkl"):
    os.makedirs(
        os.path.dirname(model_path), exist_ok=True
    )  # Tạo thư mục nếu chưa tồn tại
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path="./models/catboost_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {os.path.abspath(model_path)}. Please run main.py first to train and save the model."
        )
    return joblib.load(model_path)


# FastAPI app
app = FastAPI()

model = None

label_map = {
    0: "Normal",
    1: "Elevated",
    2: "Hypertension Stage 1",
    3: "Hypertension Stage 2",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    global model
    model = load_model()  # Load với path mặc định; có thể thay đổi nếu cần, ví dụ: load_model('../models/best_model.pkl')
    yield
    # Shutdown: Optional cleanup
    pass


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([input_data.model_dump()])
    prediction = int(model.predict(df)[0])
    label = label_map.get(prediction, "Unknown")

    return {"prediction": prediction, "label": label}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5012)

import sys
import os
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from backend.schemas import PredictionInput, PredictionOutput

# Add the project root to sys.path so we can import Model
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from Model.modelCalculations import LinearRegression

app = FastAPI(title="CPU_Usage API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Training data ──────────────────────────────────────────────
# Features: [Active_Users, Time_of_Day, Background_Jobs, DB_Latency_ms]
X_train = np.array(
    [
        [45, 2, 1, 12],
        [110, 8, 2, 15],
        [215, 10, 3, 25],
        [380, 14, 5, 40],
        [490, 16, 8, 65],
        [50, 22, 2, 10],
        [150, 9, 4, 20],
        [310, 11, 6, 35],
        [420, 15, 7, 55],
        [495, 17, 10, 80],
        [25, 4, 1, 8],
        [180, 10, 3, 22],
        [275, 12, 5, 30],
        [350, 14, 4, 45],
        [460, 18, 9, 70],
        [85, 7, 2, 18],
        [220, 13, 6, 28],
        [395, 16, 7, 50],
        [120, 21, 3, 15],
        [480, 15, 12, 90],
    ]
)

# Target: CPU_Usage_pct
y_train = np.array(
    [
        14.2,
        26.8,
        42.5,
        68.1,
        92.4,
        18.5,
        35.2,
        58.9,
        81.3,
        98.7,
        9.5,
        38.4,
        54.2,
        62.7,
        89.6,
        22.3,
        49.8,
        76.5,
        29.1,
        95.2,
    ]
)

# ── Feature normalisation (z-score) ───────────────────────────
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_norm = (X_train - X_mean) / X_std

# ── Train the model at startup ─────────────────────────────────
m, n = X_norm.shape
w_init = np.zeros(n)
b_init = 0.0
iterations = 10000
alpha = 0.01

model = LinearRegression(
    w=w_init,
    b=b_init,
    x=X_norm,
    y=y_train,
    iteration=iterations,
    alpha=alpha,
    m=m,
)
trained_w, trained_b, cost_history, iteration_history = model.linear_regression()

print(
    f"\n✅ Model trained — final weights: {np.round(trained_w, 4)}, bias: {trained_b:.4f}"
)


# ── Routes ────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root():
    """Serve the frontend page."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    """Predict CPU usage % from the given features."""
    x_input = np.array(
        [
            data.active_users,
            data.time_of_day,
            data.background_jobs,
            data.db_latency_ms,
        ]
    )
    # Apply same normalisation used during training
    x_normalised = (x_input - X_mean) / X_std
    prediction = model.predict(x_normalised)
    return PredictionOutput(
        cpu_usage_pct=round(float(prediction), 2),
        inputs=data,
    )


@app.get("/model-info")
def model_info():
    """Return trained model parameters and training metadata."""
    return {
        "weights": np.round(trained_w, 4).tolist(),
        "bias": round(float(trained_b), 4),
        "iterations": iterations,
        "alpha": alpha,
        "training_samples": m,
        "features": ["active_users", "time_of_day", "background_jobs", "db_latency_ms"],
        "target": "cpu_usage_pct",
    }

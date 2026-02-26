# CPU Usage Predictor

A full-stack machine learning application that predicts server CPU usage percentage based on real-time server metrics. Built with a **custom linear regression model from scratch** (NumPy only), a **FastAPI** backend, and a sleek **HTML/CSS/JS** frontend.

---

## Features

- **Custom Linear Regression** — Gradient descent implementation built from scratch with NumPy (no scikit-learn).
- **REST API** — FastAPI backend with endpoints for predictions and model info.
- **Interactive Frontend** — Clean, dark-themed UI with colour-coded prediction results (green/yellow/red).
- **Z-Score Normalisation** — Input features are normalised using training-set statistics for accurate predictions.

---

## Project Structure

```
CPU Usage/
├── main.py                      # Entry point — launches the FastAPI server
├── backend/
│   ├── backend.py               # FastAPI app, routes, training data & normalisation
│   └── schemas.py               # Pydantic request/response models
├── frontend/
│   ├── index.html               # HTML markup
│   ├── style.css                # Stylesheet
│   └── script.js                # Client-side JavaScript
├── Model/
│   └── modelCalculations.py     # Custom LinearRegression class (NumPy)
└── README.md
```

---

## Input Features

| Feature           | Description                        | Example |
| ----------------- | ---------------------------------- | ------- |
| `active_users`    | Number of currently active users   | 300     |
| `time_of_day`     | Hour of the day (0–23)             | 14      |
| `background_jobs` | Number of running background jobs  | 5       |
| `db_latency_ms`   | Database latency in milliseconds   | 30      |

**Output:** Predicted CPU usage as a percentage (0–100%).

---

## Getting Started

### Prerequisites

- Python 3.8+

### Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd "CPU Usage"
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\Activate.ps1

   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn numpy
   ```

### Running the Application

```bash
python main.py
```

The server starts at **http://127.0.0.1:8000**. Open that URL in your browser to use the predictor UI.

---

## API Endpoints

| Method | Endpoint       | Description                                  |
| ------ | -------------- | -------------------------------------------- |
| GET    | `/`            | Serves the frontend page                     |
| GET    | `/health`      | Health check — returns `{"status": "ok"}`    |
| POST   | `/predict`     | Predict CPU usage from input features        |
| GET    | `/model-info`  | Returns trained weights, bias & metadata     |

### Example — POST `/predict`

**Request:**
```json
{
  "active_users": 300,
  "time_of_day": 14,
  "background_jobs": 5,
  "db_latency_ms": 30
}
```

**Response:**
```json
{
  "cpu_usage_pct": 58.42,
  "inputs": {
    "active_users": 300,
    "time_of_day": 14,
    "background_jobs": 5,
    "db_latency_ms": 30
  }
}
```

---

## How It Works

1. **Training** — At server startup, the model trains on 20 labelled samples using gradient descent (10 000 iterations, learning rate 0.01).
2. **Normalisation** — Features are z-score normalised using training-set mean and standard deviation.
3. **Prediction** — User inputs are normalised with the same statistics, then passed through the trained linear model: $\hat{y} = \mathbf{w} \cdot \mathbf{x} + b$.
4. **Display** — The frontend colour-codes the result:
   - **Green** ≤ 40% — low usage
   - **Yellow** 40–75% — moderate usage
   - **Red** > 75% — high usage

---

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript (vanilla)
- **ML:** NumPy (custom implementation)

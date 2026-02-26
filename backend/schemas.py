from pydantic import BaseModel


class PredictionInput(BaseModel):
    active_users: float
    time_of_day: float
    background_jobs: float
    db_latency_ms: float


class PredictionOutput(BaseModel):
    cpu_usage_pct: float
    inputs: PredictionInput

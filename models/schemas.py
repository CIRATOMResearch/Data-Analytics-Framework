from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class DataUpload(BaseModel):
    filename: str
    content_type: str
    size: int


class AnalysisResult(BaseModel):
    analysis_type: str
    results: Dict[str, Any]
    created_at: str


class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class BasicMetrics(BaseModel):
    mean: float
    median: float
    mode: Optional[float] = None
    std: float
    var: float
    min: float
    max: float
    count: int


class DispersionMetrics(BaseModel):
    variance: float
    std_dev: float
    range: float
    iqr: float
    mad: float


class Quartiles(BaseModel):
    q1: float
    q2: float
    q3: float
    p10: float
    p90: float


class DistributionMetrics(BaseModel):
    skewness: float
    kurtosis: float
    is_normal: bool


class CorrelationResult(BaseModel):
    correlation_matrix: Dict[str, Dict[str, float]]
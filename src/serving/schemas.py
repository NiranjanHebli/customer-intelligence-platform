from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class CustomerFeatures(BaseModel):
    age: int = Field(..., ge=0)
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: int = Field(..., ge=0)
    campaign: int = Field(..., ge=1)
    pdays: int = Field(..., ge=0)
    previous: int = Field(..., ge=0)
    poutcome: str
    emp_var_rate: float = Field(..., alias="emp.var.rate")
    cons_price_idx: float = Field(..., alias="cons.price.idx")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx")
    euribor3m: float = Field(..., alias="euribor3m")
    nr_employed: float = Field(..., alias="nr.employed")

    model_config = {
        "populate_by_name": True
    }


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    threshold_decision: str
    model_version: str


class BatchPredictRequest(BaseModel):
    features: List[CustomerFeatures]


class BatchPredictResponse(BaseModel):
    total_scored: int
    conversion_counts: Dict[int, int]
    model_version: str

class AskComplaintRequest(BaseModel):
    question: str = Field(..., min_length=5)
    filter_product: Optional[str] = None

class AskComplaintResponse(BaseModel):
    answer: str
    cited_evidence_ids: List[str]
    evidence_sufficiency_note: str
    prompt_version: str

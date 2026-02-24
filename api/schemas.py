from pydantic import BaseModel, Field
from typing import List

class PredictionResponse(BaseModel):
    class_name: List[str] = Field(
        description="Categories of image"
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Probability of that class"
    )

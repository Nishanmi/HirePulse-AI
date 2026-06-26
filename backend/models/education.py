from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from backend.models.enums import Tier

class Education(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    institution: str = Field(...)
    degree: str = Field(...)
    field_of_study: str = Field(...)
    start_year: int = Field(..., ge=1970, le=2030)
    end_year: int = Field(..., ge=1970, le=2035)
    grade: Optional[str] = Field(None, description="GPA / percentage / class.")
    tier: Optional[Tier] = Field(None, description="Internal tiering for institution prestige.")

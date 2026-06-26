from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from backend.models.enums import CompanySize

class CareerHistory(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    company: str = Field(...)
    title: str = Field(...)
    start_date: date = Field(...)
    end_date: Optional[date] = Field(...)
    duration_months: int = Field(..., ge=0)
    is_current: bool = Field(...)
    industry: str = Field(...)
    company_size: CompanySize = Field(...)
    description: str = Field(..., description="Role responsibilities and achievements.")

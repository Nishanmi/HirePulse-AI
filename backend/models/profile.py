from pydantic import BaseModel, Field, ConfigDict
from backend.models.enums import CompanySize

class Profile(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    anonymized_name: str = Field(..., description="Anonymized full name.")
    headline: str = Field(..., description="One-line professional headline.")
    summary: str = Field(..., description="Multi-sentence professional summary.")
    location: str = Field(..., description="City, region/state.")
    country: str = Field(...)
    years_of_experience: float = Field(..., ge=0, le=50)
    current_title: str = Field(...)
    current_company: str = Field(...)
    current_company_size: CompanySize = Field(...)
    current_industry: str = Field(...)

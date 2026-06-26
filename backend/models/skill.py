from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from backend.models.enums import SkillProficiency

class Skill(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    name: str = Field(...)
    proficiency: SkillProficiency = Field(...)
    endorsements: int = Field(..., ge=0)
    duration_months: Optional[int] = Field(None, ge=0, description="Months the candidate has used this skill")

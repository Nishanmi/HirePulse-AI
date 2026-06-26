from pydantic import BaseModel, Field, ConfigDict
from backend.models.enums import LanguageProficiency

class Language(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    language: str = Field(...)
    proficiency: LanguageProficiency = Field(...)

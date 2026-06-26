from pydantic import BaseModel, Field, ConfigDict

class Certification(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    name: str = Field(...)
    issuer: str = Field(...)
    year: int = Field(...)

from pydantic import BaseModel, Field, ConfigDict

from backend.models.profile import Profile
from backend.models.career import CareerHistory
from backend.models.education import Education
from backend.models.skill import Skill
from backend.models.certification import Certification
from backend.models.language import Language
from backend.models.redrob_signals import RedrobSignals

class Candidate(BaseModel):
    """
    Candidate profile in the Intelligent Candidate Discovery & Ranking Challenge dataset.
    """
    model_config = ConfigDict(validate_assignment=True)
    
    candidate_id: str = Field(
        ..., 
        pattern=r"^CAND_[0-9]{7}$", 
        description="Unique identifier for the candidate. Format: CAND_XXXXXXX (7 digits)."
    )
    profile: Profile = Field(...)
    career_history: list[CareerHistory] = Field(..., min_length=1, max_length=10)
    education: list[Education] = Field(..., max_length=5)
    skills: list[Skill] = Field(...)
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    redrob_signals: RedrobSignals = Field(...)

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class CandidateFeatures(BaseModel):
    """
    Represents engineered features and scores extracted from a candidate profile.
    
    This model contains various independent signals evaluated against a job description,
    used by the ranking system to determine the final ordering of candidates.
    It does not store raw candidate data, only the derived feature values.
    """
    model_config = ConfigDict(validate_assignment=True)
    
    semantic_match_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Score representing the semantic similarity between the candidate's profile and the job description."
    )
    
    technical_match_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Score representing how well the candidate's technical skills match the required and preferred skills."
    )
    
    experience_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Score based on the candidate's years of experience compared to the role's requirements."
    )
    
    behavioral_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Score evaluating behavioral signals and soft skills mentioned in the profile."
    )
    
    recruiter_interest_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Score simulating a recruiter's overall interest based on domain, companies, and progression."
    )
    
    availability_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Score based on the candidate's notice period, location compatibility, and willingness to relocate."
    )
    
    validation_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Score reflecting the validity and consistency of the profile (lower score indicates potential anomalies)."
    )
    
    culture_fit_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Score indicating alignment with the company's culture and workplace environment preferences."
    )
    
    final_score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="The aggregated final ranking score derived from all other feature scores."
    )

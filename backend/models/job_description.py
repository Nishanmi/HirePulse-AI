from typing import Optional
from pydantic import BaseModel, Field, model_validator, ConfigDict


class ExperienceRequirement(BaseModel):
    """
    Represents the experience requirements for a role.
    """
    model_config = ConfigDict(validate_assignment=True)

    minimum_years: Optional[float] = Field(None, ge=0.0, description="The minimum years of experience required.")
    maximum_years: Optional[float] = Field(None, ge=0.0, description="The maximum years of experience expected or allowed.")

    @model_validator(mode='after')
    def validate_experience_range(self) -> 'ExperienceRequirement':
        """Validates that the experience range is logical."""
        if self.minimum_years is not None and self.maximum_years is not None:
            if self.minimum_years > self.maximum_years:
                raise ValueError("maximum_years cannot be less than minimum_years")
        return self


class JobDescription(BaseModel):
    """
    Represents a structured understanding of a job description.
    
    This model captures the essential requirements, preferences, and cultural
    aspects of a role, primarily used for downstream candidate retrieval and ranking.
    """
    model_config = ConfigDict(validate_assignment=True)

    title: str = Field(..., description="The official job title for the position.")
    company: Optional[str] = Field(None, description="The name of the hiring company.")
    
    role_embedding: Optional[list[float]] = Field(None, description="Embedding of the normalized job title.")
    normalized_role_title: Optional[str] = Field(None, description="Cleaned, normalized version of the job title.")
    
    jd_requirements_embedding: Optional[list[float]] = Field(None, description="Embedding of JD core requirements text for career-evidence matching.")
    
    preferred_locations: list[str] = Field(default_factory=list, description="List of acceptable geographical locations.")
    work_mode: Optional[str] = Field(None, description="The work mode (e.g., Remote, Hybrid, On-site).")
    relocation_required: Optional[bool] = Field(None, description="Indicates if relocation is required for this role.")
    
    employment_type: Optional[str] = Field(None, description="The type of employment (e.g., Full-time, Contract, Part-time).")
    
    experience: Optional[ExperienceRequirement] = Field(None, description="The experience requirements for the role.")
    
    must_have_skills: list[str] = Field(default_factory=list, description="Critical skills that a candidate must possess.")
    preferred_skills: list[str] = Field(default_factory=list, description="Skills that are nice to have but not strictly required.")
    
    required_domains: list[str] = Field(default_factory=list, description="Specific industries or domains the candidate must have experience in.")
    preferred_companies: list[str] = Field(default_factory=list, description="Companies from which candidates are preferred.")
    
    education_preferences: list[str] = Field(default_factory=list, description="Preferred educational backgrounds or degrees.")
    certification_preferences: list[str] = Field(default_factory=list, description="Preferred or required certifications.")
    keywords: list[str] = Field(default_factory=list, description="Important keywords extracted from the job description.")
    desired_traits: list[str] = Field(default_factory=list, description="Specific personal or professional traits desired.")
    evaluation_signals: list[str] = Field(default_factory=list, description="Key signals to look for when evaluating candidates.")
    
    responsibilities: list[str] = Field(default_factory=list, description="Key responsibilities and duties of the role.")
    behavioral_expectations: list[str] = Field(default_factory=list, description="Expected behavioral traits or soft skills (e.g., leadership, teamwork).")
    culture_preferences: list[str] = Field(default_factory=list, description="Cultural values or workplace environment preferences (e.g., fast-paced, startup).")
    disqualifiers: list[str] = Field(default_factory=list, description="Conditions or traits that explicitly disqualify a candidate.")

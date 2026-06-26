from datetime import date
from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict
from backend.models.enums import WorkMode

class SalaryRange(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    min: float = Field(..., ge=0)
    max: float = Field(..., ge=0)

class RedrobSignals(BaseModel):
    """
    Simulated platform activity and engagement signals from the Redrob ecosystem.
    """
    model_config = ConfigDict(validate_assignment=True)
    
    profile_completeness_score: float = Field(..., ge=0, le=100, description="Percentage of profile completeness.")
    signup_date: date = Field(...)
    last_active_date: date = Field(...)
    open_to_work_flag: bool = Field(...)
    profile_views_received_30d: int = Field(..., ge=0)
    applications_submitted_30d: int = Field(..., ge=0)
    recruiter_response_rate: float = Field(..., ge=0, le=1, description="Fraction of recruiter messages the candidate has responded to.")
    avg_response_time_hours: float = Field(..., ge=0)
    skill_assessment_scores: dict[str, Annotated[float, Field(ge=0, le=100)]] = Field(
        ..., 
        description="Dict of skill_name -> score 0-100. Assessments completed on Redrob platform."
    )
    connection_count: int = Field(..., ge=0)
    endorsements_received: int = Field(..., ge=0)
    notice_period_days: int = Field(..., ge=0, le=180)
    expected_salary_range_inr_lpa: SalaryRange = Field(..., description="Expected salary in INR Lakhs Per Annum.")
    preferred_work_mode: WorkMode = Field(...)
    willing_to_relocate: bool = Field(...)
    github_activity_score: float = Field(..., ge=-1, le=100, description="0-100 score based on commits, PRs, stars in last 12 months. -1 if no GitHub linked.")
    search_appearance_30d: int = Field(..., ge=0, description="Number of times profile appeared in recruiter searches in last 30 days.")
    saved_by_recruiters_30d: int = Field(..., ge=0, description="Number of recruiters who saved this profile in last 30 days.")
    interview_completion_rate: float = Field(..., ge=0, le=1, description="Fraction of scheduled interviews actually attended.")
    offer_acceptance_rate: float = Field(..., ge=-1, le=1, description="Historical offer acceptance rate. -1 if no offer history.")
    verified_email: bool = Field(...)
    verified_phone: bool = Field(...)
    linkedin_connected: bool = Field(...)

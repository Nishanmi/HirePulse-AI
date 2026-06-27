import re
from typing import Set

from backend.models import Candidate, JobDescription, CandidateFeatures


class FeatureEngine:
    """
    Extracts comparable features and computes deterministic match scores
    between a Candidate and a JobDescription.
    """

    def extract_features(self, candidate: Candidate, jd: JobDescription) -> CandidateFeatures:
        """
        Computes deterministic feature scores for a candidate against a job description.
        
        Args:
            candidate (Candidate): The candidate profile.
            jd (JobDescription): The structured job description.
            
        Returns:
            CandidateFeatures: Extracted feature scores.
        """
        return CandidateFeatures(
            semantic_match_score=self._compute_semantic_match(candidate, jd),
            technical_match_score=self._compute_technical_match(candidate, jd),
            experience_score=self._compute_experience_match(candidate, jd),
            behavioral_score=self._compute_behavioral_match(candidate, jd),
            recruiter_interest_score=self._compute_recruiter_interest(candidate),
            availability_score=self._compute_availability(candidate, jd),
            validation_score=self._compute_validation_score(candidate),
            culture_fit_score=self._compute_culture_fit(candidate, jd),
            final_score=None  # Explicitly leaving final_score out of feature extraction
        )

    def _extract_words(self, text: str) -> Set[str]:
        """Helper to extract unique lower-case words from text."""
        if not text:
            return set()
        words = re.findall(r'\b\w+\b', text.lower())
        return set(words)

    def _get_candidate_text(self, candidate: Candidate) -> str:
        """Aggregates significant text fields from a candidate profile."""
        parts = [
            candidate.profile.headline,
            candidate.profile.summary
        ]
        for role in candidate.career_history:
            parts.extend([role.title, role.description])
        return " ".join(filter(None, parts))

    def _compute_semantic_match(self, candidate: Candidate, jd: JobDescription) -> float:
        """Heuristic semantic match using Jaccard-like similarity of keywords."""
        candidate_text = self._get_candidate_text(candidate)
        jd_text = " ".join(filter(None, [
            jd.title,
            " ".join(jd.responsibilities),
            " ".join(jd.must_have_skills),
            " ".join(jd.preferred_skills)
        ]))

        candidate_words = self._extract_words(candidate_text)
        jd_words = self._extract_words(jd_text)
        
        # Filter out common stop words to improve heuristic precision
        stop_words = {"and", "or", "the", "a", "an", "in", "on", "to", "for", "with", "is", "of", "experience", "skills"}
        candidate_words -= stop_words
        jd_words -= stop_words

        if not jd_words:
            return 0.5
            
        intersection = candidate_words.intersection(jd_words)
        
        # Calculate coverage of JD keywords found in Candidate profile
        coverage = len(intersection) / len(jd_words)
        # Boost score slightly to allow a reasonable distribution between 0 and 1
        return min(1.0, coverage * 1.5)

    def _compute_technical_match(self, candidate: Candidate, jd: JobDescription) -> float:
        """Matches candidate skills against JD must_have and preferred skills."""
        if not jd.must_have_skills and not jd.preferred_skills:
            return 1.0

        # Create a set of lowercased candidate skill names for O(1) lookups
        candidate_skills = {s.name.lower() for s in candidate.skills}
        
        must_have_matches = 0
        preferred_matches = 0

        # We do simple substring checks to be a bit more robust than exact matching
        for jd_skill in jd.must_have_skills:
            if any(jd_skill.lower() in cand_skill or cand_skill in jd_skill.lower() for cand_skill in candidate_skills):
                must_have_matches += 1
                
        for jd_skill in jd.preferred_skills:
            if any(jd_skill.lower() in cand_skill or cand_skill in jd_skill.lower() for cand_skill in candidate_skills):
                preferred_matches += 1

        must_have_score = 0.0
        if jd.must_have_skills:
            must_have_score = must_have_matches / len(jd.must_have_skills)
            
        preferred_score = 0.0
        if jd.preferred_skills:
            preferred_score = preferred_matches / len(jd.preferred_skills)

        # Base skill score
        if jd.must_have_skills and jd.preferred_skills:
            skill_score = (must_have_score * 0.7) + (preferred_score * 0.3)
        elif jd.must_have_skills:
            skill_score = must_have_score
        else:
            skill_score = preferred_score
            
        # Title mismatch penalty (Honeypot trap filter)
        title_penalty = 1.0
        if jd.title and candidate.profile.current_title:
            jd_title_words = self._extract_words(jd.title)
            cand_title_words = self._extract_words(candidate.profile.current_title)
            
            # If there's absolutely zero overlap in words between JD title and Candidate title
            if jd_title_words and cand_title_words and not jd_title_words.intersection(cand_title_words):
                # Slash the technical match score by 80% to penalize keyword stuffers with irrelevant titles
                title_penalty = 0.2
                
        return skill_score * title_penalty

    def _compute_experience_match(self, candidate: Candidate, jd: JobDescription) -> float:
        """Compares candidate's years of experience with JD requirements."""
        if not jd.experience:
            return 1.0
            
        cand_exp = candidate.profile.years_of_experience
        min_exp = jd.experience.minimum_years or 0.0
        max_exp = jd.experience.maximum_years or float('inf')

        if cand_exp < min_exp:
            # Penalize linearly for being under-experienced
            return max(0.0, cand_exp / min_exp) if min_exp > 0 else 1.0
        elif cand_exp > max_exp:
            # Slight penalty for being over-qualified
            over = cand_exp - max_exp
            penalty = min(0.5, over * 0.05)
            return 1.0 - penalty
        else:
            # Perfectly within range
            return 1.0

    def _compute_behavioral_match(self, candidate: Candidate, jd: JobDescription) -> float:
        """Matches behavioral expectations against candidate profile and signals."""
        base_score = 0.5
        signals = candidate.redrob_signals
        
        # Boost based on positive platform signals representing professional behavior
        if signals.recruiter_response_rate > 0.8:
            base_score += 0.1
        if signals.interview_completion_rate > 0.8:
            base_score += 0.1
            
        # Keyword matching for soft skills
        if jd.behavioral_expectations:
            candidate_words = self._extract_words(self._get_candidate_text(candidate))
            jd_behavioral_words = self._extract_words(" ".join(jd.behavioral_expectations))
            
            if jd_behavioral_words:
                intersection = candidate_words.intersection(jd_behavioral_words)
                coverage = len(intersection) / len(jd_behavioral_words)
                base_score += (coverage * 0.3)
                
        # SEVERE BEHAVIORAL PENALTIES (Hackathon constraint)
        # 1. Response Rate Trap
        if signals.recruiter_response_rate < 0.10:
            base_score *= 0.1  # 90% penalty for ghosting recruiters
            
        # 2. Inactive/Stale Trap
        from datetime import date
        days_inactive = (date.today() - signals.last_active_date).days
        if days_inactive > 90:
            base_score *= 0.1  # 90% penalty if hasn't logged in for 3 months
                
        return min(1.0, max(0.0, base_score))

    def _compute_recruiter_interest(self, candidate: Candidate) -> float:
        """Uses Redrob signals to determine how interesting the candidate is to recruiters generally."""
        signals = candidate.redrob_signals
        
        # Normalize various signals to 0-1 and aggregate
        views_score = min(1.0, signals.profile_views_received_30d / 50.0)
        saves_score = min(1.0, signals.saved_by_recruiters_30d / 10.0)
        connection_score = min(1.0, signals.connection_count / 500.0)
        
        # Weight them to form a final proxy score
        score = (views_score * 0.4) + (saves_score * 0.4) + (connection_score * 0.2)
        return min(1.0, max(0.0, score))

    def _compute_availability(self, candidate: Candidate, jd: JobDescription) -> float:
        """Scores candidate based on notice period, relocation, and open-to-work flag."""
        signals = candidate.redrob_signals
        score = 0.5
        
        if signals.open_to_work_flag:
            score += 0.2
            
        # Notice period: shorter is better
        if signals.notice_period_days <= 15:
            score += 0.2
        elif signals.notice_period_days <= 30:
            score += 0.1
        elif signals.notice_period_days > 60:
            score -= 0.2
            
        # Relocation checks
        if jd.relocation_required and signals.willing_to_relocate:
            score += 0.1
        elif jd.relocation_required and not signals.willing_to_relocate:
            score -= 0.3
            
        return min(1.0, max(0.0, score))

    def _compute_validation_score(self, candidate: Candidate) -> float:
        """Scores candidate based on profile completeness and verifications, penalizing anomalies."""
        signals = candidate.redrob_signals
        score = signals.profile_completeness_score / 100.0
        
        # Penalize missing verifications slightly
        if not signals.verified_email:
            score -= 0.1
        if not signals.verified_phone:
            score -= 0.1
            
        # TIMELINE ANOMALY TRAP FILTER (e.g. 10 years exp but graduated 2 years ago)
        yoe = candidate.profile.years_of_experience
        latest_grad_year = 0
        for edu in candidate.education:
            if edu.end_year and edu.end_year > latest_grad_year:
                latest_grad_year = edu.end_year
                
        if latest_grad_year > 0:
            from datetime import date
            current_year = date.today().year
            # Max possible post-grad experience (allowing 3 years of leeway for working during school)
            max_realistic_yoe = max(0, current_year - latest_grad_year) + 3
            if yoe > max_realistic_yoe:
                # Severe penalty for impossible timeline
                score *= 0.1
                
        return min(1.0, max(0.0, score))

    def _compute_culture_fit(self, candidate: Candidate, jd: JobDescription) -> float:
        """Scores culture fit using basic keyword matching against culture preferences."""
        if not jd.culture_preferences:
            return 0.5
            
        candidate_words = self._extract_words(self._get_candidate_text(candidate))
        culture_words = self._extract_words(" ".join(jd.culture_preferences))
        
        if not culture_words:
            return 0.5
            
        intersection = candidate_words.intersection(culture_words)
        coverage = len(intersection) / len(culture_words)
        
        # Boost the coverage factor slightly to give a reasonable distribution
        return min(1.0, coverage * 2.0)

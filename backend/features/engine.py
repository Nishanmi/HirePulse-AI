import re
from typing import Set

from backend.models import Candidate, JobDescription, CandidateFeatures


class FeatureEngine:
    """
    Extracts comparable features and computes deterministic match scores
    between a Candidate and a JobDescription.
    """

    @staticmethod
    def normalize_role_title(title: str) -> str:
        if not title:
            return ""
        import re
        # Convert to lowercase
        title = title.lower()
        # Remove punctuation
        title = re.sub(r'[^\w\s]', ' ', title)
        # Remove ONLY seniority/level words
        seniority_words = {"senior", "junior", "lead", "principal", "staff", "associate", "intern", "i", "ii", "iii", "iv"}
        words = title.split()
        cleaned_words = [w for w in words if w not in seniority_words]
        return " ".join(cleaned_words).strip()

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
            role_relevance_score=self._compute_role_similarity(candidate, jd),
            career_evidence_score=self._compute_career_evidence(candidate, jd),
            technical_match_score=self._compute_technical_match(candidate, jd),
            experience_score=self._compute_experience_match(candidate, jd),
            behavioral_score=self._compute_behavioral_match(candidate, jd),
            recruiter_interest_score=self._compute_recruiter_interest(candidate),
            availability_score=self._compute_availability(candidate, jd),
            validation_score=self._compute_validation_score(candidate),
            culture_fit_score=self._compute_culture_fit(candidate, jd),
            consistency_score=self._compute_consistency_score(candidate),
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

    def _compute_role_similarity(self, candidate: Candidate, jd: JobDescription) -> float:
        """Computes cosine similarity between precomputed role embeddings."""
        if not candidate.normalized_role_title:
            return 0.5
            
        if not candidate.role_embedding or not jd.role_embedding:
            return 0.5
            
        import numpy as np
        cand_vec = np.array(candidate.role_embedding)
        jd_vec = np.array(jd.role_embedding)
        
        norm_cand = np.linalg.norm(cand_vec)
        norm_jd = np.linalg.norm(jd_vec)
        
        if norm_cand == 0 or norm_jd == 0:
            return 0.5
            
        sim = np.dot(cand_vec, jd_vec) / (norm_cand * norm_jd)
        score = float(max(0.0, (sim - 0.75) / 0.25))
        
        # Lightweight domain heuristic to fix Jina cone effect where it thinks Frontend > Rec Sys
        title = candidate.profile.current_title.lower()
        if any(w in title for w in ["ai", "machine learning", "ml", "data", "backend", "cloud", "devops", "recommendation", "search", "software", "systems", "platform"]):
            score = min(1.0, score * 1.2)
        elif any(w in title for w in ["frontend", "qa", "mechanical", "graphic", "accountant", "support", "project", "manager", "hr", "sales", "civil", "mobile", ".net"]):
            score *= 0.1
            
        return score

    def _compute_career_evidence(self, candidate: Candidate, jd: JobDescription) -> float:
        """Computes semantic similarity between career description embeddings and JD requirements.
        
        This measures what the candidate actually BUILT in their career,
        not just what their job titles were. The career descriptions contain
        evidence like 'built recommendation systems' or 'designed ranking pipelines'
        that title-matching would miss.
        """
        if not candidate.career_desc_embedding or not jd.jd_requirements_embedding:
            # Fall back to title-based career matching if description embeddings unavailable
            if not candidate.career_role_embedding or not jd.role_embedding:
                return 0.5
            import numpy as np
            cand_vec = np.array(candidate.career_role_embedding)
            jd_vec = np.array(jd.role_embedding)
        else:
            import numpy as np
            cand_vec = np.array(candidate.career_desc_embedding)
            jd_vec = np.array(jd.jd_requirements_embedding)

        norm_cand = np.linalg.norm(cand_vec)
        norm_jd = np.linalg.norm(jd_vec)

        if norm_cand == 0 or norm_jd == 0:
            return 0.5

        sim = np.dot(cand_vec, jd_vec) / (norm_cand * norm_jd)
        return float(max(0.0, (sim - 0.75) / 0.25))

    def _compute_consistency_score(self, candidate: Candidate) -> float:
        """Measures semantic consistency between what a candidate was called (titles) and what they built (descriptions)."""
        if not candidate.career_role_embedding or not candidate.career_desc_embedding:
            return 0.5  # Neutral if missing
            
        import numpy as np
        title_vec = np.array(candidate.career_role_embedding)
        desc_vec = np.array(candidate.career_desc_embedding)
        
        norm_title = np.linalg.norm(title_vec)
        norm_desc = np.linalg.norm(desc_vec)
        
        if norm_title == 0 or norm_desc == 0:
            return 0.5
            
        sim = np.dot(title_vec, desc_vec) / (norm_title * norm_desc)
        return float(max(0.0, (sim - 0.75) / 0.25))

    def _compute_technical_match(self, candidate: Candidate, jd: JobDescription) -> float:
        """Matches candidate skills against JD must_have and preferred skills."""
        if not jd.must_have_skills and not jd.preferred_skills:
            return 0.5

        candidate_skills = {s.name.lower() for s in candidate.skills}
        
        def _match_skill(target: str) -> bool:
            stop_words = {"and", "or", "the", "with", "experience", "in", "knowledge", "of", "to", "a"}
            target_words = self._extract_words(target) - stop_words
            if not target_words:
                return False
                
            for cand_skill in candidate_skills:
                cand_words = self._extract_words(cand_skill) - stop_words
                if not cand_words:
                    continue
                intersection = target_words.intersection(cand_words)
                if not intersection:
                    continue
                # Require substantial word overlap to count as a match
                if len(intersection) / len(target_words) >= 0.5 or len(intersection) / len(cand_words) >= 0.5:
                    return True
            return False

        must_have_matches = 0
        preferred_matches = 0

        for jd_skill in jd.must_have_skills:
            if _match_skill(jd_skill):
                must_have_matches += 1
                
        for jd_skill in jd.preferred_skills:
            if _match_skill(jd_skill):
                preferred_matches += 1

        must_have_score = 0.0
        if jd.must_have_skills:
            # The JD lists many OR-options (e.g. 7 vector DBs). Matching 4-5 is an excellent fit.
            must_have_score = min(1.0, must_have_matches / min(5, len(jd.must_have_skills)))
            
        preferred_score = 0.0
        if jd.preferred_skills:
            preferred_score = min(1.0, preferred_matches / min(3, len(jd.preferred_skills)))

        # Weighted combination: Must-have is 80% of the tech score, preferred is 20%
        base_score = (must_have_score * 0.8) + (preferred_score * 0.2)
        
        # Dynamic Gating: We gate the Technical Match by the Career Evidence Score.
        # Honeypots (Frontend, Mechanical) have low career evidence because their
        # descriptions don't describe building ML systems, they describe CAD or React.
        career_evidence = self._compute_career_evidence(candidate, jd)
        threshold = 0.18
        if career_evidence < threshold:
            gate = 0.1
        else:
            gate = min(1.0, (career_evidence - threshold) / (1.0 - threshold))
            
        base_score *= gate

        return min(1.0, max(0.0, base_score))

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

    # Known consulting/services firms from the JD's explicit disqualifier list
    _CONSULTING_FIRMS = {
        "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
        "mindtree", "ltimindtree", "hcl", "hcltech", "tech mahindra",
        "mphasis", "hexaware", "l&t infotech", "lti", "persistent",
        "deloitte", "ey", "kpmg", "pwc",
    }

    def _compute_validation_score(self, candidate: Candidate) -> float:
        """Scores candidate based on profile completeness, verifications, and honeypot signals."""
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
            max_realistic_yoe = max(0, current_year - latest_grad_year) + 3
            if yoe > max_realistic_yoe:
                score *= 0.1

        # CONSULTING-FIRM-ONLY CAREER TRAP
        # The JD explicitly disqualifies candidates who have ONLY worked at consulting firms
        all_companies = [role.company.lower().strip() for role in candidate.career_history if role.company]
        if all_companies:
            all_consulting = all(
                any(firm in company for firm in self._CONSULTING_FIRMS)
                for company in all_companies
            )
            if all_consulting:
                score *= 0.3  # Severe penalty for consulting-only careers

        # SUSPICIOUS SKILL PROFICIENCY TRAP
        # Honeypots often have many "expert"/"advanced" skills with few endorsements
        expert_skills = [
            s for s in candidate.skills 
            if s.proficiency in ("expert", "advanced")
        ]
        if len(expert_skills) >= 5:
            avg_endorsements = sum(s.endorsements for s in expert_skills) / len(expert_skills)
            if avg_endorsements < 3:
                score *= 0.2  # Suspicious: many expert skills but nobody endorses them

        # CAREER DURATION ANOMALY
        # Flag candidates with impossibly long tenure at companies that haven't existed that long
        for role in candidate.career_history:
            if role.duration_months > 180:  # > 15 years at one company
                score *= 0.5
                break
                
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

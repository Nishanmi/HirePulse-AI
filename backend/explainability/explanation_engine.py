import logging
from typing import Dict, List, Optional, Set, Tuple

from backend.models.candidate import Candidate
from backend.models.candidate_features import CandidateFeatures
from backend.models.enums import SkillProficiency
from backend.models.job_description import JobDescription

logger = logging.getLogger(__name__)

# Skill synonyms for fuzzy matching (lowercase -> canonical form)
_SKILL_SYNONYMS: Dict[str, str] = {
    "reactjs": "react",
    "react.js": "react",
    "react js": "react",
    "nodejs": "node.js",
    "node": "node.js",
    "node js": "node.js",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "pytorch": "pytorch",
    "torch": "pytorch",
    "postgres": "postgresql",
    "psql": "postgresql",
    "k8s": "kubernetes",
    "kube": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "python3": "python",
    "aws": "aws",
    "amazon web services": "aws",
    "gcp": "google cloud",
    "google cloud platform": "google cloud",
    "ml": "machine learning",
    "nlp": "natural language processing",
    "dl": "deep learning",
    "ci/cd": "ci/cd",
    "cicd": "ci/cd",
    "mongo": "mongodb",
    "elastic search": "elasticsearch",
    "fastapi": "fastapi",
    "fast api": "fastapi",
}

# Common filler words to exclude when extracting skills from JD phrases
_FILLER_WORDS: Set[str] = {
    "strong", "solid", "good", "excellent", "deep", "proficiency", "proficient",
    "experience", "understanding", "knowledge", "familiarity", "familiar",
    "with", "in", "of", "the", "a", "an", "to", "for", "on", "at", "by",
    "is", "are", "be", "been", "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "can", "may", "might", "must",
    "not", "no", "yes", "this", "that", "these", "those", "it", "its",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "than", "too", "very", "just", "also", "only",
    "own", "same", "so", "about", "up", "out", "if", "into", "over",
    "after", "before", "between", "under", "above", "below", "from",
    "any", "many", "much", "well", "like", "building", "using", "working",
    "ability", "able", "required", "preferred", "minimum", "maximum",
    "years", "year", "months", "month", "modern", "stacks", "techniques",
    "algorithms", "data", "science", "systems", "tools",
}


class ExplanationEngine:
    """
    Generates deterministic, factual 1-2 sentence explanations for why a
    candidate was ranked and matched to a job description.

    Explanations are built from concrete data present in the candidate profile,
    extracted features, and the job description. No information is invented.
    The engine uses skill synonym matching to accurately credit candidates
    for skill variations.

    Priority order:
        1. Matched JD skills (must-have and preferred)
        2. Experience alignment with current role context
        3. Recruiter interest (brief summary)
        4. Validation concerns (only when they affected the score)
    """

    def explain(
        self,
        candidate: Candidate,
        features: CandidateFeatures,
        jd: JobDescription,
    ) -> str:
        """
        Generates a concise 1-2 sentence explanation for the candidate's ranking.

        Args:
            candidate: The Candidate domain model.
            features: The engineered features/scores for this candidate.
            jd: The JobDescription domain model.

        Returns:
            A single explanation string of 1-2 well-formed sentences.
        """
        # Build the primary sentence (skills + experience + role)
        primary = self._build_primary_sentence(candidate, features, jd)

        # Build the secondary sentence (recruiter interest + validation)
        secondary = self._build_secondary_sentence(candidate, features)

        if primary and secondary:
            return f"{primary} {secondary}"
        if primary:
            return primary
        if secondary:
            return secondary

        return (
            "Candidate shows moderate potential based on an aggregation "
            "of baseline profile signals."
        )

    # ------------------------------------------------------------------
    # Private helpers — sentence builders
    # ------------------------------------------------------------------

    def _build_primary_sentence(
        self,
        candidate: Candidate,
        features: CandidateFeatures,
        jd: JobDescription,
    ) -> Optional[str]:
        """Builds the primary sentence combining matched skills, experience,
        and current role into one cohesive statement."""
        fragments: List[str] = []

        # Matched skills
        cand_skill_map = self._get_candidate_skill_map(candidate)
        cand_canonical = self._canonicalise_set(cand_skill_map.keys())
        must_have_matched = self._match_skills_from_phrases(
            jd.must_have_skills, cand_canonical
        )
        preferred_matched = self._match_skills_from_phrases(
            jd.preferred_skills, cand_canonical
        )
        all_matched = must_have_matched + preferred_matched

        if all_matched:
            skills_str = self._format_skill_list(all_matched, cand_skill_map)
            fragments.append(f"matches key skills including {skills_str}")

        # Experience + current role
        years = candidate.profile.years_of_experience
        title = candidate.profile.current_title
        industry = candidate.profile.current_industry

        role_ctx = ""
        if title and industry:
            role_ctx = f" as {title} in {industry}"
        elif title:
            role_ctx = f" as {title}"

        if features.experience_score is not None and features.experience_score > 0.8:
            exp_fragment = f"brings {years:.0f} years of experience{role_ctx}"
            if jd.experience and jd.experience.minimum_years is not None:
                if jd.experience.maximum_years is not None:
                    exp_fragment += (
                        f", well within the {jd.experience.minimum_years:.0f}"
                        f"-{jd.experience.maximum_years:.0f} year requirement"
                    )
            fragments.append(exp_fragment)
        elif features.experience_score is not None and features.experience_score > 0.5:
            fragments.append(
                f"has {years:.0f} years of experience{role_ctx}, adequate for the role"
            )
        elif years is not None:
            fragments.append(f"has {years:.0f} years of experience{role_ctx}")

        if not fragments:
            return None

        # Join fragments into a single well-formed sentence
        sentence = fragments[0].capitalize()
        if len(fragments) > 1:
            sentence += ", and " + fragments[1]
        return sentence + "."

    def _build_secondary_sentence(
        self, candidate: Candidate, features: CandidateFeatures
    ) -> Optional[str]:
        """Builds the secondary sentence covering recruiter interest and
        validation concerns briefly."""
        parts: List[str] = []

        # Brief recruiter interest summary
        if (
            features.recruiter_interest_score is not None
            and features.recruiter_interest_score > 0.75
        ):
            signals = candidate.redrob_signals
            if signals.open_to_work_flag:
                parts.append("Shows strong recruiter interest and is actively open to new opportunities.")
            else:
                parts.append("Shows strong recruiter engagement on the platform.")

        # Validation concerns (only when they affected the score)
        if features.validation_score is not None and features.validation_score < 0.4:
            parts.append(
                "However, profile validation flagged concerns that negatively impacted the ranking."
            )

        if not parts:
            return None

        return " ".join(parts)

    def _get_candidate_skill_map(
        self, candidate: Candidate
    ) -> Dict[str, SkillProficiency]:
        """Returns a mapping of lowercase skill name -> proficiency."""
        return {
            s.name.strip().lower(): s.proficiency for s in candidate.skills
        }

    def _match_skills_from_phrases(
        self, jd_skills: List[str], candidate_canonical: Set[str]
    ) -> List[str]:
        """Matches individual skill keywords extracted from JD phrases
        against a candidate's canonicalised skill set.

        JD skills are often full phrases like 'Strong proficiency in Python
        and modern data science stacks'. This method extracts individual
        keywords (e.g. 'Python') and checks them against the candidate.
        """
        matched: List[str] = []
        seen: Set[str] = set()

        for phrase in jd_skills:
            # First try exact match on the whole phrase
            canonical = self._canonicalise(phrase.strip().lower())
            if canonical in candidate_canonical and canonical not in seen:
                matched.append(phrase.strip())
                seen.add(canonical)
                continue

            # Extract individual skill tokens from the phrase
            for token in self._extract_skills_from_phrase(phrase):
                token_canonical = self._canonicalise(token.lower())
                if token_canonical in candidate_canonical and token_canonical not in seen:
                    matched.append(token)
                    seen.add(token_canonical)

        return matched

    def _extract_skills_from_phrase(self, phrase: str) -> List[str]:
        """Extracts individual skill-like tokens from a JD requirement phrase.

        Splits on common delimiters (commas, 'and', 'or', slashes) and filters
        out generic filler words to isolate meaningful skill names.
        """
        import re

        # Split on commas, 'and', 'or', slashes, parentheses
        tokens = re.split(r"[,/()]", phrase)
        tokens = [t for part in tokens for t in re.split(r"\band\b|\bor\b", part)]

        skills: List[str] = []
        for token in tokens:
            cleaned = token.strip().rstrip(".")
            if not cleaned:
                continue
            # Check if the whole token is a known skill or synonym
            if self._canonicalise(cleaned.lower()) in _SKILL_SYNONYMS.values():
                skills.append(cleaned)
                continue
            if cleaned.lower() in _SKILL_SYNONYMS:
                skills.append(cleaned)
                continue
            # Extract individual words that might be skills
            for word in cleaned.split():
                word_clean = word.strip().rstrip(".")
                word_lower = word_clean.lower()
                if len(word_clean) <= 1:
                    continue
                canonical = self._canonicalise(word_lower)
                # Accept if it's in the synonym map or is capitalised/technical
                if (
                    word_lower in _SKILL_SYNONYMS
                    or canonical != word_lower
                    or (len(word_clean) >= 2 and word_clean[0].isupper() and word_lower not in _FILLER_WORDS)
                ):
                    skills.append(word_clean)

        return skills

    def _extract_keywords_from_phrases(self, phrases: List[str]) -> List[str]:
        """Extracts all unique skill keywords from a list of JD phrases."""
        seen: Set[str] = set()
        keywords: List[str] = []
        for phrase in phrases:
            for skill in self._extract_skills_from_phrase(phrase):
                canonical = self._canonicalise(skill.lower())
                if canonical not in seen:
                    seen.add(canonical)
                    keywords.append(skill)
        return keywords

    def _format_skill_list(
        self,
        skills: List[str],
        cand_skill_map: Dict[str, SkillProficiency],
    ) -> str:
        """Formats matched skills with proficiency annotations where available."""
        formatted: List[str] = []
        for skill in skills[:4]:
            prof = cand_skill_map.get(skill.lower())
            if prof and prof in (SkillProficiency.ADVANCED, SkillProficiency.EXPERT):
                formatted.append(f"{skill} ({prof.value})")
            else:
                formatted.append(skill)

        if len(skills) > 4:
            formatted.append(f"and {len(skills) - 4} others")

        if len(formatted) > 1:
            return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"
        return formatted[0]

    def _canonicalise(self, skill: str) -> str:
        """Returns the canonical form of a skill name using the synonym map."""
        return _SKILL_SYNONYMS.get(skill, skill)

    def _canonicalise_set(self, skills: Set[str]) -> Set[str]:
        """Returns the canonical forms of a set of skill names."""
        return {self._canonicalise(s) for s in skills}

import logging
import re
from typing import Dict, List

from rank_bm25 import BM25Okapi

from backend.models import Candidate, JobDescription
from backend.retrieval.retriever import RetrievalStrategy

logger = logging.getLogger(__name__)


class BM25Strategy(RetrievalStrategy):
    """
    BM25-based lexical retrieval strategy.

    Builds an inverted index from candidate profile text and scores
    candidates against a tokenised query derived from the JobDescription.
    Scores are normalised to [0, 1].
    """

    def score_candidates(
        self,
        candidates: List[Candidate],
        jd: JobDescription,
    ) -> Dict[str, float]:
        """
        Scores each candidate using BM25 relevance against the job description.

        Args:
            candidates: The full pool of candidates.
            jd: The structured job description.

        Returns:
            A dict mapping candidate_id to a normalised relevance score in [0, 1].
        """
        if not candidates:
            return {}

        corpus = [self._build_candidate_document(c) for c in candidates]
        query = self._build_query(jd)

        if not query:
            logger.warning("BM25Strategy: empty query derived from job description.")
            return {c.candidate_id: 0.0 for c in candidates}

        tokenised_corpus = [self._tokenise(doc) for doc in corpus]
        tokenised_query = self._tokenise(query)

        bm25 = BM25Okapi(tokenised_corpus)
        raw_scores = bm25.get_scores(tokenised_query)

        scores = self._normalise_scores(raw_scores)

        return {
            candidates[i].candidate_id: scores[i]
            for i in range(len(candidates))
        }

    def _build_candidate_document(self, candidate: Candidate) -> str:
        """Assembles a single text document from all relevant candidate fields."""
        parts: List[str] = [
            candidate.profile.headline,
            candidate.profile.summary,
            candidate.profile.current_title,
            candidate.profile.current_company,
            candidate.profile.current_industry,
        ]

        for role in candidate.career_history:
            parts.extend([role.title, role.company, role.industry, role.description])

        for edu in candidate.education:
            parts.extend([edu.institution, edu.degree, edu.field_of_study])

        for skill in candidate.skills:
            parts.append(skill.name)

        for cert in candidate.certifications:
            parts.append(cert.name)

        return " ".join(filter(None, parts))

    def _build_query(self, jd: JobDescription) -> str:
        """Assembles a query string from the most relevant JD fields."""
        parts: List[str] = [jd.title]

        parts.extend(jd.must_have_skills)
        parts.extend(jd.preferred_skills)
        parts.extend(jd.required_domains)
        parts.extend(jd.responsibilities)
        parts.extend(jd.keywords)

        return " ".join(filter(None, parts))

    def _tokenise(self, text: str) -> List[str]:
        """Lowercases text and splits into word tokens, stripping non-alphanumerics."""
        return re.findall(r"\b\w+\b", text.lower())

    def _normalise_scores(self, raw_scores) -> List[float]:
        """Min-max normalises BM25 scores to the [0, 1] range."""
        if len(raw_scores) == 0:
            return []

        max_score = float(max(raw_scores))
        min_score = float(min(raw_scores))
        score_range = max_score - min_score

        if score_range == 0:
            return [0.0] * len(raw_scores)

        return [
            (float(score) - min_score) / score_range
            for score in raw_scores
        ]

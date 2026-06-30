import re
from typing import List, Optional

from backend.models import JobDescription, ExperienceRequirement


class JDExtractor:
    """
    Extracts structured job description data from raw plain text.
    """

    def extract(self, text: str) -> JobDescription:
        """
        Parses plain text job description and returns a validated JobDescription model.
        
        Args:
            text (str): The raw job description text.
            
        Returns:
            JobDescription: A validated Pydantic model.
            
        Raises:
            ValueError: If the input text is empty or invalid.
        """
        if not text or not text.strip():
            raise ValueError("Job description text cannot be empty.")

        title = self._extract_title(text)
        company = self._extract_company(text)
        location = self._extract_location(text)
        employment_type = self._extract_employment_type(text)
        
        min_years = self._extract_experience_min(text)
        max_years = self._extract_experience_max(text)

        must_have_items = self._extract_section_items(text, ["things you absolutely need", "must have", "requirements", "required skills", "qualifications"])
        must_have_skills = self._extract_skill_names_from_section(must_have_items)
        preferred_items = self._extract_section_items(text, ["things we'd like you to have", "preferred", "nice to have", "bonus"])
        preferred_skills = self._extract_skill_names_from_section(preferred_items)
        required_domains = self._extract_section_items(text, ["domain", "industry experience"])
        preferred_companies = self._extract_section_items(text, ["preferred companies", "worked at"])
        responsibilities = self._extract_section_items(text, ["responsibilities", "what you'll do", "what you'd actually be doing", "day to day"])
        behavioral_expectations = self._extract_section_items(text, ["behavioral", "soft skills", "expectations"])
        culture_preferences = self._extract_section_items(text, ["culture", "values", "environment"])
        disqualifiers = self._extract_section_items(text, ["things we explicitly do not want", "disqualifiers", "not a fit if", "dealbreakers"])

        preferred_locations = [location] if location else []
        experience = None
        if min_years is not None or max_years is not None:
            # Ensure max_years >= min_years to pass validation
            if min_years is not None and max_years is not None and max_years < min_years:
                max_years = min_years
            experience = ExperienceRequirement(minimum_years=min_years, maximum_years=max_years)

        return JobDescription(
            title=title or "Untitled Role",
            company=company,
            preferred_locations=preferred_locations,
            employment_type=employment_type,
            experience=experience,
            must_have_skills=must_have_skills,
            preferred_skills=preferred_skills,
            required_domains=required_domains,
            preferred_companies=preferred_companies,
            responsibilities=responsibilities,
            behavioral_expectations=behavioral_expectations,
            culture_preferences=culture_preferences,
            disqualifiers=disqualifiers
        )

    def _extract_title(self, text: str) -> str:
        """Extracts the job title."""
        match = re.search(r"(?i)^(?:title|role|position):\s*([^\n]+)", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        # Fallback: assume first non-empty line might be the title if it's short
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines and len(lines[0]) < 100:
            return lines[0]
        return "Untitled Role"

    def _extract_company(self, text: str) -> Optional[str]:
        """Extracts the company name."""
        match = re.search(r"(?i)^(?:company|organization):\s*([^\n]+)", text, re.MULTILINE)
        return match.group(1).strip() if match else None

    def _extract_location(self, text: str) -> Optional[str]:
        """Extracts the location."""
        match = re.search(r"(?i)^(?:location|based in):\s*([^\n]+)", text, re.MULTILINE)
        return match.group(1).strip() if match else None

    def _extract_employment_type(self, text: str) -> Optional[str]:
        """Extracts the employment type."""
        match = re.search(r"(?i)^(?:employment type|type|job type):\s*([^\n]+)", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
        
        # Look for common keywords
        lower_text = text.lower()
        if "full-time" in lower_text or "full time" in lower_text:
            return "Full-time"
        elif "part-time" in lower_text or "part time" in lower_text:
            return "Part-time"
        elif "contract" in lower_text:
            return "Contract"
        
        return None

    def _extract_experience_min(self, text: str) -> Optional[float]:
        """Extracts the minimum years of experience."""
        # Looks for "X+ years" or "X to Y years"
        match = re.search(r"(?i)(\d+)(?:\+|-| to \d+)?\s*years?(?: of)? experience", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None

    def _extract_experience_max(self, text: str) -> Optional[float]:
        """Extracts the maximum years of experience."""
        # Looks for "X to Y years"
        match = re.search(r"(?i)(?:\d+\s*(?:-|to)\s*)(\d+)\s*years?(?: of)? experience", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None

    def _extract_section_items(self, text: str, section_keywords: List[str]) -> List[str]:
        """
        Extracts list items from a section identified by keywords.

        Handles sections with bullet characters (•, -, *, numbered items),
        tab-indented content, and sections that flow into the next section
        header without double newlines.
        """
        # Build a pattern that matches any known section header line.
        # This is used to detect section boundaries.
        _all_section_keywords = [
            "things you absolutely need",
            "things we'd like you to have",
            "things we explicitly do not want",
            "must have", "requirements", "required skills", "qualifications",
            "preferred", "nice to have", "bonus",
            "domain", "industry experience",
            "preferred companies", "worked at",
            "responsibilities", "what you'll do",
            "what you'd actually be doing", "day to day",
            "behavioral", "soft skills", "expectations",
            "culture", "values", "environment",
            "disqualifiers", "not a fit if", "dealbreakers",
            "on location", "the vibe check",
            "let's be honest", "what we mean by",
            "the skills inventory",
        ]

        for keyword in section_keywords:
            # Build regex: keyword can appear anywhere on a line, optionally
            # followed by a colon, and possibly preceded by other text on the
            # same line (e.g., "The skills inventory (please read carefully)").
            escaped = re.escape(keyword)
            pattern = rf"(?im)^[^\n]*{escaped}[^\n]*$"
            header_match = re.search(pattern, text)
            if not header_match:
                continue

            # Content starts on the line after the header
            content_start = header_match.end()
            remaining = text[content_start:]

            # Find where this section ends: at the next section header or EOF.
            # A "section header" is a non-bullet, non-empty line whose text
            # matches one of the known section keywords (case-insensitive).
            content_lines: List[str] = []
            for line in remaining.split('\n'):
                stripped = line.strip()
                if not stripped:
                    # Empty lines within bullet content are allowed; keep
                    # collecting unless we already have items.
                    if content_lines:
                        # Check if remaining content has more bullets
                        content_lines.append('')
                    continue

                # Is this line a bullet item?
                is_bullet = bool(re.match(r'^[•\-\*]|\d+[.\)]\s', stripped))

                if not is_bullet and content_lines:
                    # Check if this line looks like a new section header
                    line_lower = stripped.lower()
                    is_new_section = any(
                        kw in line_lower for kw in _all_section_keywords
                        if kw != keyword.lower()
                    )
                    if is_new_section:
                        break

                content_lines.append(line)

            # Parse bullet items from collected content
            items: List[str] = []
            for line in content_lines:
                stripped = line.strip()
                if not stripped:
                    continue
                # Match bullet prefixes: •, -, *, or numbered (1. / 1))
                bullet_match = re.match(r'^(?:[•\-\*]|\d+[.\)])\s*', stripped)
                if bullet_match:
                    item_text = stripped[bullet_match.end():].strip()
                    if item_text:
                        items.append(item_text)
                elif items:
                    # Continuation line for the previous bullet
                    items[-1] += ' ' + stripped
                else:
                    # Non-bullet content before any bullets
                    items.append(stripped)

            if items:
                return items

        return []

    def _extract_skill_names_from_section(self, bullet_items: List[str]) -> List[str]:
        """
        Extracts individual skill and technology names from bullet paragraphs.

        Handles:
        - Parenthetical lists: ``(sentence-transformers, OpenAI embeddings, BGE, E5, or similar)``
        - Dash/em-dash separated lists: ``— Pinecone, Weaviate, Qdrant``
        - Standalone technology mentions: ``Python``, ``NDCG``, ``MRR``, ``MAP``
        - Fine-tuning shorthand: ``(LoRA, QLoRA, PEFT)``

        Args:
            bullet_items: List of full bullet text strings from a JD section.

        Returns:
            Deduplicated list of extracted skill/technology names.
        """
        # Known filler words to strip from comma-separated lists
        _filler = {
            'or', 'and', 'similar', 'something similar', 'or similar',
            'etc', 'etc.', 'based', 'based or neural',
        }

        skills: List[str] = []
        seen: set[str] = set()

        def _add(name: str) -> None:
            """Add a skill name if it's non-trivial and not already seen."""
            name = name.strip(' ,.')
            if not name or name.lower() in _filler or len(name) < 2:
                return
            key = name.lower()
            if key not in seen:
                seen.add(key)
                skills.append(name)

        for item in bullet_items:
            # 1. Extract from parenthetical lists: (A, B, C, or similar)
            paren_matches = re.findall(r'\(([^)]+)\)', item)
            for paren_content in paren_matches:
                parts = [p.strip() for p in paren_content.split(',')]
                for part in parts:
                    # Remove trailing filler like "or similar"
                    cleaned = re.sub(r'\s+or\s+(?:something\s+)?similar$', '', part, flags=re.IGNORECASE).strip()
                    # Handle "OpenAI embeddings" -> extract "embeddings" as a skill
                    # but keep compound names that are actual tech names
                    if cleaned:
                        _add(cleaned)

            # 2. Extract from dash/em-dash separated lists:
            #    "— Pinecone, Weaviate, Qdrant, ..." 
            dash_match = re.search(r'(?:\s[—–-]\s)(.+?)(?:\.|$)', item)
            if dash_match:
                dash_content = dash_match.group(1)
                # Only treat as a list if it contains commas
                if ',' in dash_content:
                    parts = [p.strip() for p in dash_content.split(',')]
                    for part in parts:
                        cleaned = re.sub(r'\s+or\s+(?:something\s+)?similar$', '', part, flags=re.IGNORECASE).strip()
                        if cleaned:
                            _add(cleaned)

            # 3. Extract standalone well-known technology/metric names
            # These are uppercase acronyms or known tech names in context
            standalone_patterns = [
                r'\bPython\b',
                r'\bNDCG\b', r'\bMRR\b', r'\bMAP\b',
                r'\bLoRA\b', r'\bQLoRA\b', r'\bPEFT\b',
                r'\bXGBoost\b',
                r'\bBM25\b',
                r'\bFAISS\b',
            ]
            for pat in standalone_patterns:
                if re.search(pat, item):
                    match = re.search(pat, item)
                    if match:
                        _add(match.group(0))

        return skills

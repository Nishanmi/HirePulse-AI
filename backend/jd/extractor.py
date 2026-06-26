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

        must_have_skills = self._extract_section_items(text, ["must have", "requirements", "required skills", "qualifications"])
        preferred_skills = self._extract_section_items(text, ["preferred", "nice to have", "bonus"])
        required_domains = self._extract_section_items(text, ["domain", "industry experience"])
        preferred_companies = self._extract_section_items(text, ["preferred companies", "worked at"])
        responsibilities = self._extract_section_items(text, ["responsibilities", "what you'll do", "day to day"])
        behavioral_expectations = self._extract_section_items(text, ["behavioral", "soft skills", "expectations"])
        culture_preferences = self._extract_section_items(text, ["culture", "values", "environment"])
        disqualifiers = self._extract_section_items(text, ["disqualifiers", "not a fit if", "dealbreakers"])

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
        Finds the section header and extracts bullet points until the next empty line.
        """
        for keyword in section_keywords:
            # Look for keyword on its own line or followed by a colon
            pattern = rf"(?i)^{keyword}s?[:]?\s*\n([\s\S]*?)(?:\n\s*\n|\Z)"
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                content = match.group(1)
                # Split by newline and look for bullet points
                items = []
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                        items.append(line.lstrip('-*• \t'))
                    elif line and not items:
                        # If no bullets yet, just add the line
                        items.append(line)
                
                if items:
                    return items
                    
        return []

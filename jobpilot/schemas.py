"""Pydantic models shared by all tools (see plan Section 6).

Every tool returns one of these models, so LLM output is always validated
before the rest of the app touches it. Field descriptions are included in the
JSON schema we send to the LLM, so they double as extraction instructions.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Seniority = Literal["intern", "junior", "mid", "senior", "unknown"]
Language = Literal["en", "ms", "mixed"]
ExperienceFit = Literal["below", "meets", "above"]
ApplicationStatus = Literal["saved", "applied", "interview", "rejected", "offer"]


# --- F1: Job description -----------------------------------------------------


class JobPosting(BaseModel):
    title: str = Field(description="Job title as written in the posting.")
    company: str | None = Field(default=None, description="Hiring company, if stated.")
    location: str | None = Field(default=None, description="City/country or 'Remote', if stated.")
    seniority: Seniority = Field(default="unknown", description="Level implied by title and requirements.")
    required_skills: list[str] = Field(
        default_factory=list,
        description="Must-have skills/tools, short canonical names (e.g. 'Python', 'Docker').",
    )
    nice_to_have_skills: list[str] = Field(
        default_factory=list,
        description="Preferred/bonus skills, short canonical names.",
    )
    min_years_experience: float | None = Field(
        default=None, description="Minimum years of experience required, if stated."
    )
    salary_min_myr: int | None = Field(default=None, description="Monthly minimum salary in MYR, if stated.")
    salary_max_myr: int | None = Field(default=None, description="Monthly maximum salary in MYR, if stated.")
    language: Language = Field(default="en", description="Language of the posting.")
    red_flags: list[str] = Field(
        default_factory=list,
        description="Warning signs, e.g. 'junior role asking 5 yrs exp', 'unpaid', 'vague duties'.",
    )


# --- F2: Resume --------------------------------------------------------------


class Project(BaseModel):
    name: str
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    link: str | None = None


class Experience(BaseModel):
    title: str
    company: str
    start: str | None = Field(default=None, description="Start date as written, e.g. 'Jan 2024'.")
    end: str | None = Field(default=None, description="End date as written, or 'Present'.")
    highlights: list[str] = Field(default_factory=list, description="Bullet points, copied faithfully.")


class Education(BaseModel):
    institution: str
    degree: str | None = None
    field: str | None = None
    graduation_year: int | None = None


class ResumeProfile(BaseModel):
    name: str
    skills: list[str] = Field(default_factory=list, description="All skills/tools mentioned anywhere.")
    projects: list[Project] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    years_experience: float = Field(
        default=0.0, ge=0, description="Total professional experience in years (internships included)."
    )


# --- F3: Matching ------------------------------------------------------------


class SkillMatch(BaseModel):
    """One JD skill mapped to an equivalent resume skill (or None)."""

    jd_skill: str
    resume_skill: str | None = Field(
        default=None,
        description="The resume skill that means the same thing, or null if none does.",
    )


class SkillNormalisation(BaseModel):
    """LLM output for the synonym-normalisation step inside the matcher."""

    matches: list[SkillMatch]


class MatchResult(BaseModel):
    score: int = Field(ge=0, le=100)
    required_skill_coverage: float = Field(ge=0, le=1)
    nice_skill_coverage: float = Field(ge=0, le=1)
    experience_fit: ExperienceFit
    matched_skills: list[str]
    missing_skills: list[str]
    explanation: str


# --- F7: Tracker -------------------------------------------------------------


class Application(BaseModel):
    id: int | None = None
    company: str
    role: str
    score: int = Field(ge=0, le=100)
    status: ApplicationStatus = "saved"
    date_saved: date = Field(default_factory=date.today)
    date_applied: date | None = None
    notes: str = ""

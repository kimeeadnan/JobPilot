"""All prompts in one place.

Keeping prompts here (not scattered in tools) makes them easy to review,
version and compare during evaluation. Templates use str.format placeholders.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

# --- Structured-output plumbing (used by llm/client.py) ----------------------


def json_instructions(schema: type[BaseModel]) -> str:
    """Tell the model exactly which JSON shape to return."""
    return (
        "Respond with ONLY a single JSON object (no markdown, no commentary) "
        "that validates against this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )


RETRY_TEMPLATE = """\
Your previous answer was not valid for the required schema.

Previous answer:
{previous}

Validation errors:
{errors}

Return a corrected JSON object only."""


# --- F1: JD parser -----------------------------------------------------------

JD_PARSER_SYSTEM = """\
You extract structured data from job descriptions.
Rules:
- Only extract what the posting states or clearly implies. Never guess a company, salary or location.
- Skills: short canonical names ("Python", "PyTorch", "REST APIs"), no duplicates, no soft skills \
unless explicitly required.
- Put a skill in required_skills only if the posting says it is required/must-have; \
"preferred", "bonus", "a plus" go to nice_to_have_skills.
- Salary: monthly MYR integers. If an annual MYR figure is given, divide by 12. \
If no salary or a non-MYR currency, use null.
- The posting may be in English, Bahasa Malaysia, or mixed. Output skill names in English.
- red_flags: concrete warning signs only (e.g. "junior role asking 5+ years", "unpaid", \
"commission only", "very vague duties"). Empty list if none."""

JD_PARSER_USER = """\
Job description:
\"\"\"
{jd_text}
\"\"\""""


# --- F2: Resume parser -------------------------------------------------------

RESUME_PARSER_SYSTEM = """\
You extract structured data from a resume.
Rules:
- Copy facts faithfully. Do not add, embellish or infer skills that are not written.
- skills: every technical skill/tool mentioned anywhere (skills section, projects, experience), \
short canonical names, no duplicates.
- years_experience: total paid work including internships, in years (e.g. 0.5 for 6 months). \
Do not count university projects.
- Keep experience highlights as written."""

RESUME_PARSER_USER = """\
Resume text:
\"\"\"
{resume_text}
\"\"\""""


# --- F3: Matcher -------------------------------------------------------------

SKILL_NORMALISER_SYSTEM = """\
You decide whether skills mean the same thing.
For each job skill, return the resume skill that is the same skill or a direct alias \
(e.g. "torch" = "PyTorch", "JS" = "JavaScript", "Postgres" = "PostgreSQL"), otherwise null.
Do NOT match related-but-different skills (e.g. "TensorFlow" is not "PyTorch", \
"AWS" is not "GCP"). Be strict."""

SKILL_NORMALISER_USER = """\
Job skills:
{jd_skills}

Resume skills:
{resume_skills}

Return one entry in "matches" for every job skill, in the same order."""

MATCH_EXPLAINER_SYSTEM = """\
You explain a job match score to a job seeker in plain, honest language.
The score was already computed; do not change or re-estimate it.
Write 3-4 sentences: what fits, what is missing, and whether it is worth applying."""

MATCH_EXPLAINER_USER = """\
Role: {title}
Score: {score}/100
Required skill coverage: {required_coverage:.0%}
Nice-to-have coverage: {nice_coverage:.0%}
Experience fit: {experience_fit} (role asks {required_years} years, candidate has {candidate_years})
Matched skills: {matched}
Missing skills: {missing}"""

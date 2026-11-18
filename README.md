# JobPilot

An AI agent that helps with my own job search: it reads a job description, compares it with my resume, scores the fit, finds skill gaps, drafts a tailored cover letter, and logs every application in a tracker.

> 🚧 Work in progress. Week 1: foundations and parsers.

## Planned features

- **JD Parser**: job description → structured JSON (skills, seniority, salary, red flags)
- **Resume Parser**: PDF/text → structured profile
- **Match Scorer**: hybrid score (Python skill overlap + LLM explanation)
- **Gap Analyzer, Cover Letter Writer, Resume Tweaks**: evidence-only, never invents experience
- **Screening Checker**: requirement-by-requirement A–D grade instead of a fake "ATS score"
- **Application Tracker + Weekly Insights**: SQLite
- **Streamlit UI** deployed on Hugging Face Spaces

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your GEMINI_API_KEY
pytest
```

## Ethics

No scraping of LinkedIn/JobStreet, no auto-applying. I review and send everything myself.

import importlib
import os
from pathlib import Path


def _resolve_dotenv():
    try:
        module = importlib.import_module("dotenv")
        return getattr(module, "load_dotenv")
    except ModuleNotFoundError:  # pragma: no cover
        def _noop(*args, **kwargs):
            return False

        return _noop


load_dotenv = _resolve_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DOTENV_PATH = BASE_DIR / ".env"
if DOTENV_PATH.exists():
    load_dotenv(DOTENV_PATH)
else:  # pragma: no cover
    load_dotenv()


class Config:
    DEBUG = True
    TESTING = False
    SECRET_KEY = os.getenv("SECRET_KEY", "dev")
    PDF_PATH = os.getenv(
        "PDF_PATH",
        "/Users/timgoerner/Desktop/HNW_aq/Investor Personas + Case Studies.pdf",
    )
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:4200,http://127.0.0.1:4200")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
    LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
    LINKEDIN_REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI")
    FRESH_LINKEDIN_API_KEY = os.getenv("FRESH_LINKEDIN_API_KEY") or os.getenv("RAPID_API_KEY")
    FRESH_LINKEDIN_API_HOST = os.getenv(
        "FRESH_LINKEDIN_API_HOST", os.getenv("RAPID_API_HOST", "fresh-linkedin-scraper-api.p.rapidapi.com")
    )
    LINKEDIN_ORG_ID = os.getenv("LINKEDIN_ORG_ID")
    MAX_LINKEDIN_RESULTS = int(os.getenv("MAX_LINKEDIN_RESULTS", "5"))
    GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
    GOOGLE_SEARCH_CX = os.getenv("GOOGLE_SEARCH_CX")

    # Session cookie settings for local dev; adjust for production.
    # For local dev on 127.0.0.1: use Lax and not secure (HTTP)
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_DOMAIN = os.getenv("SESSION_COOKIE_DOMAIN", "127.0.0.1")
    # Optional integrations
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
    LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
    LINKEDIN_REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI")
    FRESH_LINKEDIN_API_KEY = os.getenv("FRESH_LINKEDIN_API_KEY") or os.getenv("RAPID_API_KEY")
    FRESH_LINKEDIN_API_HOST = os.getenv(
        "FRESH_LINKEDIN_API_HOST", os.getenv("RAPID_API_HOST", "fresh-linkedin-scraper-api.p.rapidapi.com")
    )
    MAX_LINKEDIN_RESULTS = int(os.getenv("MAX_LINKEDIN_RESULTS", "5"))

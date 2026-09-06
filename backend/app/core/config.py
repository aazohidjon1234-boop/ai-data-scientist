"""Application settings loaded from environment variables / .env file.

Secrets (LLM API keys, database credentials) are ALWAYS read from the
environment — never hard-coded and never exposed to the frontend.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App -----------------------------------------------------------
    app_name: str = "AI Data Scientist Agent API"
    app_version: str = "1.0.0"
    database_url: str = f"sqlite:///{BACKEND_DIR / 'data' / 'app.db'}"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # --- Storage ---------------------------------------------------------
    # Project layout: <root>/datasets (uploads + samples), <root>/reports, <root>/models
    upload_dir: str = str(BACKEND_DIR.parent / "datasets")
    report_dir: str = str(BACKEND_DIR.parent / "reports")
    model_dir: str = str(BACKEND_DIR.parent / "models")

    # --- Upload limits (security) ----------------------------------------
    max_upload_mb: int = 100
    max_rows: int = 1_000_000
    max_columns: int = 150
    max_cardinality: int = 50  # categorical columns with more uniques are dropped
    max_train_rows: int = 50_000  # datasets above this are sampled for training
    train_test_ratio: float = 0.2
    random_state: int = 42
    preview_rows: int = 100

    # --- Optional LLM (OpenAI-compatible) for interpretation -------------
    # When absent, a deterministic local explanation engine is used instead.
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def upload_subdir(self) -> Path:
        return Path(self.upload_dir) / "uploads"

    @property
    def sample_subdir(self) -> Path:
        return Path(self.upload_dir) / "samples"


@lru_cache
def get_settings() -> Settings:
    return Settings()

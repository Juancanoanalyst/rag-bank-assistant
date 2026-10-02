from pathlib import Path

import pytest

from rag_assistant.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"
SITE = "https://www.banco-ejemplo.com.co"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def read_fixture():
    def _read(relative_path: str) -> str:
        return (FIXTURES / relative_path).read_text(encoding="utf-8")

    return _read


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointing at a fictional site and at temporary data folders."""
    return Settings(
        _env_file=None,
        scraper_base_url=SITE,
        scraper_sitemap_url=f"{SITE}/sitemap.xml",
        scraper_delay_seconds=0,
        scraper_user_agent="test-agent",
        raw_data_dir=tmp_path / "raw",
        clean_data_dir=tmp_path / "clean",
    )

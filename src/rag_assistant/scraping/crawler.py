"""Polite sitemap-driven crawler: robots.txt, rate limit, honest user agent."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

from rag_assistant.config import Settings
from rag_assistant.exceptions import ScrapingError
from rag_assistant.scraping.robots import RobotsPolicy
from rag_assistant.scraping.sitemap import collect_urls
from rag_assistant.scraping.storage import RawStore
from rag_assistant.scraping.urls import spread_by_section

logger = logging.getLogger(__name__)

_SKIPPED_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".zip",
    ".xml", ".doc", ".docx", ".xls", ".xlsx", ".mp4",
)  # fmt: skip

# Upper bound on sitemap entries read before choosing which pages to download.
_MAX_SITEMAP_URLS = 20_000


@dataclass
class CrawlReport:
    saved: int = 0
    cached: int = 0
    failed: int = 0
    disallowed: int = 0


class SiteCrawler:
    """Downloads up to SCRAPER_MAX_PAGES pages listed in the site's sitemap."""

    def __init__(
        self,
        settings: Settings,
        store: RawStore,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._store = store
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = settings.scraper_user_agent
        self._sleep = sleep
        self._host = urlparse(settings.scraper_base_url).netloc

    def crawl(self) -> CrawlReport:
        """Fetch robots.txt and the sitemap, then download the allowed pages.

        Pages already present in the raw store are not downloaded again, so the
        command can be re-run (or resumed after an interruption) cheaply.
        """
        settings = self._settings
        robots = RobotsPolicy.fetch(
            settings.scraper_base_url,
            settings.scraper_user_agent,
            self._session,
            settings.scraper_timeout_seconds,
        )
        if not robots.allows(settings.scraper_sitemap_url):
            raise ScrapingError(
                f"robots.txt does not allow crawling {settings.scraper_base_url} "
                "(or the site refused to serve it). Save pages manually into "
                f"{settings.raw_data_dir} and run the clean step instead."
            )

        report = CrawlReport()
        allowed = []
        for url in collect_urls(settings.scraper_sitemap_url, self._get_text, _MAX_SITEMAP_URLS):
            if not self._is_candidate(url):
                continue
            if robots.allows(url):
                allowed.append(url)
            else:
                report.disallowed += 1

        selected = spread_by_section(allowed, settings.scraper_max_pages)
        already_stored = self._store.urls()
        delay = max(settings.scraper_delay_seconds, robots.crawl_delay)
        logger.info("Sitemap lists %d allowed pages; fetching %d", len(allowed), len(selected))

        for position, url in enumerate(selected, start=1):
            if url in already_stored:
                report.cached += 1
                continue

            self._sleep(delay)
            try:
                self._store.save(url, self._get_html(url))
                report.saved += 1
                logger.info("[%d/%d] saved %s", position, len(selected), url)
            except ScrapingError as exc:
                report.failed += 1
                logger.warning("[%d/%d] skipped: %s", position, len(selected), exc)

        logger.info(
            "Crawl finished: %d saved, %d already stored, %d failed, %d disallowed by robots.txt",
            report.saved,
            report.cached,
            report.failed,
            report.disallowed,
        )
        return report

    def _is_candidate(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.netloc == self._host and not parsed.path.lower().endswith(_SKIPPED_EXTENSIONS)

    def _get(self, url: str) -> requests.Response:
        try:
            response = self._session.get(url, timeout=self._settings.scraper_timeout_seconds)
        except requests.RequestException as exc:
            raise ScrapingError(f"Request to {url} failed: {exc}") from exc
        if response.status_code != 200:
            raise ScrapingError(f"{url} returned HTTP {response.status_code}")
        return response

    def _get_text(self, url: str) -> str:
        return self._get(url).text

    def _get_html(self, url: str) -> str:
        response = self._get(url)
        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type:
            raise ScrapingError(f"{url} is not HTML (Content-Type: {content_type or 'missing'})")
        return response.text

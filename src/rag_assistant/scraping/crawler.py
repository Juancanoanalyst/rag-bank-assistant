"""Polite sitemap-driven crawler: robots.txt, rate limit, honest user agent."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests

from rag_assistant.config import Settings
from rag_assistant.exceptions import ScrapingError
from rag_assistant.scraping.robots import RobotsPolicy
from rag_assistant.scraping.sitemap import collect_urls
from rag_assistant.scraping.storage import RawStore, decode_html
from rag_assistant.scraping.urls import sample_evenly

logger = logging.getLogger(__name__)

_SKIPPED_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".zip",
    ".xml", ".doc", ".docx", ".xls", ".xlsx", ".mp4",
)  # fmt: skip

# Upper bound on sitemap entries read before choosing which pages to download.
_MAX_SITEMAP_URLS = 20_000
_MAX_REDIRECTS = 5
# Hammering a site that answers 403/429 to everything would be impolite.
_MAX_CONSECUTIVE_FAILURES = 10


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

        selected = sample_evenly(allowed, settings.scraper_max_pages)
        already_stored = self._store.urls()
        delay = max(settings.scraper_delay_seconds, robots.crawl_delay)
        logger.info("Sitemap lists %d allowed pages; fetching %d", len(allowed), len(selected))

        consecutive_failures = 0
        for position, url in enumerate(selected, start=1):
            if url in already_stored:
                report.cached += 1
                continue
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.error(
                    "Stopping: %d pages failed in a row, the site may be rejecting this client",
                    consecutive_failures,
                )
                break

            self._sleep(delay)
            try:
                self._store.save(*self._get_html(url, robots))
                report.saved += 1
                consecutive_failures = 0
                logger.info("[%d/%d] saved %s", position, len(selected), url)
            except ScrapingError as exc:
                report.failed += 1
                consecutive_failures += 1
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

    def _request(self, url: str, allow_redirects: bool) -> requests.Response:
        try:
            return self._session.get(
                url,
                timeout=self._settings.scraper_timeout_seconds,
                allow_redirects=allow_redirects,
            )
        except requests.RequestException as exc:
            raise ScrapingError(f"Request to {url} failed: {exc}") from exc

    def _get_text(self, url: str) -> str:
        response = self._request(url, allow_redirects=True)
        if response.status_code != 200:
            raise ScrapingError(f"{url} returned HTTP {response.status_code}")
        return response.text

    def _get_html(self, url: str, robots: RobotsPolicy) -> tuple[str, str]:
        """Download a page and return (final_url, html).

        Redirects are followed by hand so that every hop is checked against the
        host filter and robots.txt: a sitemap URL must not smuggle in a
        disallowed path or another site.
        """
        for _ in range(_MAX_REDIRECTS + 1):
            response = self._request(url, allow_redirects=False)
            if not response.is_redirect:
                break
            url = urljoin(url, response.headers["Location"])
            if not self._is_candidate(url) or not robots.allows(url):
                raise ScrapingError(f"Redirect to {url} leaves the allowed scope")
        else:
            raise ScrapingError(f"Too many redirects, last one to {url}")

        if response.status_code != 200:
            raise ScrapingError(f"{url} returned HTTP {response.status_code}")
        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type:
            raise ScrapingError(f"{url} is not HTML (Content-Type: {content_type or 'missing'})")
        # Without an explicit charset, requests assumes ISO-8859-1 and garbles accents.
        if "charset=" in content_type.lower():
            return url, response.text
        return url, decode_html(response.content)

"""robots.txt handling."""

import logging
from urllib.parse import urljoin

import requests
from protego import Protego

from rag_assistant.exceptions import ScrapingError

logger = logging.getLogger(__name__)


class RobotsPolicy:
    """Answers "may this user agent fetch this URL?" for one site.

    Parsing is delegated to protego because the standard library parser ignores
    wildcard rules such as `Disallow: *.content.html`, which real sites use.
    """

    def __init__(self, parser: Protego, user_agent: str) -> None:
        self._parser = parser
        self._user_agent = user_agent

    @classmethod
    def from_text(cls, text: str, user_agent: str) -> "RobotsPolicy":
        return cls(Protego.parse(text), user_agent)

    @classmethod
    def fetch(
        cls, base_url: str, user_agent: str, session: requests.Session, timeout: float
    ) -> "RobotsPolicy":
        """Download and parse robots.txt.

        A missing file (404) means everything is allowed. A 401/403 is read as
        "disallow everything": if the site refuses to show its rules to this
        client, we do not crawl it.
        """
        robots_url = urljoin(base_url, "/robots.txt")
        try:
            response = session.get(robots_url, timeout=timeout)
        except requests.RequestException as exc:
            raise ScrapingError(f"Could not fetch {robots_url}: {exc}") from exc

        if response.status_code in (401, 403):
            logger.warning(
                "%s returned %s: treating the site as disallowed", robots_url, response.status_code
            )
            return cls.from_text("User-agent: *\nDisallow: /", user_agent)
        if response.status_code == 404:
            return cls.from_text("", user_agent)
        if response.status_code != 200:
            raise ScrapingError(f"{robots_url} returned HTTP {response.status_code}")
        return cls.from_text(response.text, user_agent)

    def allows(self, url: str) -> bool:
        return self._parser.can_fetch(url, self._user_agent)

    @property
    def crawl_delay(self) -> float:
        delay = self._parser.crawl_delay(self._user_agent)
        return float(delay) if delay else 0.0

"""sitemap.xml parsing (plain url sets and sitemap indexes)."""

import logging
from collections.abc import Callable
from xml.etree import ElementTree

from rag_assistant.exceptions import ScrapingError

logger = logging.getLogger(__name__)

# A sitemap index may point to other indexes; stop following them at this depth.
MAX_INDEX_DEPTH = 2


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_sitemap(xml_text: str) -> tuple[list[str], list[str]]:
    """Return (page_urls, child_sitemap_urls) found in one sitemap document."""
    try:
        root = ElementTree.fromstring(xml_text.strip())
    except ElementTree.ParseError as exc:
        raise ScrapingError(f"Sitemap is not valid XML: {exc}") from exc

    locs = [
        element.text.strip()
        for element in root.iter()
        if _local_name(element.tag) == "loc" and element.text and element.text.strip()
    ]
    if _local_name(root.tag) == "sitemapindex":
        return [], locs
    return locs, []


def collect_urls(sitemap_url: str, fetch_text: Callable[[str], str], limit: int) -> list[str]:
    """Walk a sitemap (following indexes) and return up to `limit` unique page URLs.

    `fetch_text` downloads one sitemap and raises ScrapingError on failure. A
    child sitemap that fails is skipped; a failing root sitemap aborts.
    """
    urls: dict[str, None] = {}  # insertion-ordered set
    pending = [(sitemap_url, 0)]
    seen_sitemaps = set()

    while pending and len(urls) < limit:
        current, depth = pending.pop(0)
        if current in seen_sitemaps:
            continue
        seen_sitemaps.add(current)

        try:
            pages, children = parse_sitemap(fetch_text(current))
        except ScrapingError:
            if depth == 0:
                raise
            logger.warning("Skipping child sitemap %s", current, exc_info=True)
            continue

        urls.update(dict.fromkeys(pages))
        if depth < MAX_INDEX_DEPTH:
            pending.extend((child, depth + 1) for child in children)

    return list(urls)[:limit]

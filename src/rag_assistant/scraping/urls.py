"""URL helpers shared by the crawler and the cleaner."""

from collections.abc import Iterable
from itertools import zip_longest
from urllib.parse import urlparse

# How many leading path segments describe the site section (e.g. personas/productos).
SECTION_DEPTH = 2
HOME_SECTION = "inicio"


def section_from_url(url: str) -> str:
    """Derive a section label from the URL path: /personas/cuentas/x.html -> personas/cuentas."""
    segments = [segment for segment in urlparse(url).path.split("/") if segment]
    folders = segments[:-1] if segments and "." in segments[-1] else segments
    return "/".join(folders[:SECTION_DEPTH]) or HOME_SECTION


def spread_by_section(urls: Iterable[str], limit: int) -> list[str]:
    """Pick up to `limit` URLs, taking turns between site sections.

    A sitemap lists one section after another, so "the first N URLs" would index
    a single corner of the site. Round-robin keeps small sections complete and
    splits the remaining budget between the large ones.
    """
    by_section: dict[str, list[str]] = {}
    for url in urls:
        by_section.setdefault(section_from_url(url), []).append(url)

    selected = [
        url for turn in zip_longest(*by_section.values()) for url in turn if url is not None
    ]
    return selected[:limit]

"""URL helpers shared by the crawler and the cleaner."""

from urllib.parse import urlparse

# How many leading path segments describe the site section (e.g. personas/productos).
SECTION_DEPTH = 2
HOME_SECTION = "inicio"


def section_from_url(url: str) -> str:
    """Derive a section label from the URL path: /personas/cuentas/x.html -> personas/cuentas."""
    segments = [segment for segment in urlparse(url).path.split("/") if segment]
    folders = segments[:-1] if segments and "." in segments[-1] else segments
    return "/".join(folders[:SECTION_DEPTH]) or HOME_SECTION


def sample_evenly(urls: list[str], limit: int) -> list[str]:
    """Pick up to `limit` URLs at regular intervals along the list.

    A sitemap lists one section after another, so "the first N URLs" would index
    a single corner of the site. Evenly spaced picks cover every section in
    proportion to its size, and the choice is deterministic between runs.
    """
    if len(urls) <= limit:
        return list(urls)
    return [urls[index * len(urls) // limit] for index in range(limit)]

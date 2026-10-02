"""Turn raw HTML into clean documents with trafilatura."""

import logging
from collections.abc import Iterable, Iterator

import trafilatura

from rag_assistant.models import Document, RawPage
from rag_assistant.scraping.urls import section_from_url

logger = logging.getLogger(__name__)

# Cookie banners sit outside <nav>/<footer>, so trafilatura keeps their text.
# Drop any element whose id or class mentions cookies before extracting.
_LOWER = "translate({}, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')"
_COOKIE_BANNER_XPATH = (
    f"//*[contains({_LOWER.format('@id')}, 'cookie')"
    f" or contains({_LOWER.format('@class')}, 'cookie')]"
)

# Lines the extractor leaves behind for empty list items.
_EMPTY_LINES = {"", "-", "•", "|"}


def _tidy(text: str) -> str:
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line not in _EMPTY_LINES)


class HtmlCleaner:
    """Extracts the main text of a page, dropping navigation, footers and scripts."""

    def __init__(self, min_text_chars: int) -> None:
        self._min_text_chars = min_text_chars

    def clean(self, page: RawPage) -> Document | None:
        """Return the cleaned document, or None when the page has no usable text."""
        try:
            extracted = trafilatura.bare_extraction(
                page.html,
                url=page.url,
                with_metadata=True,
                include_comments=False,
                include_tables=True,
                favor_recall=True,
                prune_xpath=_COOKIE_BANNER_XPATH,
            )
        except Exception:  # trafilatura wraps lxml; a broken page must not stop the batch
            logger.warning("Could not parse %s", page.filename, exc_info=True)
            return None

        text = _tidy(extracted.text or "") if extracted else ""
        if len(text) < self._min_text_chars:
            logger.info("Skipping %s: only %d characters of text", page.filename, len(text))
            return None

        return Document(
            url=page.url,
            title=(extracted.title or "").strip() or page.url,
            section=section_from_url(page.url),
            text=text,
        )

    def clean_all(self, pages: Iterable[RawPage]) -> Iterator[Document]:
        """Clean many pages, keeping one document per URL."""
        seen = set()
        for page in pages:
            if page.url in seen:
                logger.info("Skipping %s: duplicate of %s", page.filename, page.url)
                continue
            document = self.clean(page)
            if document is not None:
                seen.add(page.url)
                yield document

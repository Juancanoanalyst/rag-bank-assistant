"""Local storage for raw HTML (data/raw) and clean documents (data/clean)."""

import hashlib
import html as html_lib
import json
import logging
import os
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from rag_assistant.exceptions import ScrapingError
from rag_assistant.models import Document, RawPage

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.jsonl"
DOCUMENTS_NAME = "documents.jsonl"

# Chrome and Edge add this comment to every page saved with "Save as".
_SAVED_FROM = re.compile(r"<!--\s*saved from url=\(\d+\)(\S+)\s*-->", re.IGNORECASE)
# (?<![-\w]) keeps data-href / data-content from being read as the attribute.
_CANONICAL = re.compile(
    r"<link\b(?=[^>]*\brel=[\"']?canonical)[^>]*(?<![-\w])href=[\"']([^\"']+)", re.IGNORECASE
)
_OG_URL = re.compile(
    r"<meta\b(?=[^>]*\bproperty=[\"']og:url)[^>]*(?<![-\w])content=[\"']([^\"']+)",
    re.IGNORECASE,
)


def infer_url(html: str) -> str | None:
    """Recover the original URL of a page saved by hand from a browser."""
    for pattern in (_CANONICAL, _OG_URL, _SAVED_FROM):
        match = pattern.search(html)
        if match:
            url = html_lib.unescape(match.group(1))
            if url.startswith(("http://", "https://")):
                return url
    return None


def decode_html(raw: bytes) -> str:
    """Decode HTML bytes whose charset is not declared by a trustworthy header.

    UTF-8 first (what virtually every current site serves); Windows-1252 as the
    fallback, since it is what legacy Spanish-language pages and old "Save as"
    dialogs produce and it can decode any byte sequence.
    """
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _read_jsonl(path: Path) -> Iterator[dict]:
    """Yield the JSON objects of a .jsonl file, skipping lines cut short by an interrupted run."""
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Ignoring corrupt line %d of %s", number, path)


def _filename_for(url: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", url.lower().split("://", 1)[-1]).strip("-")[:80]
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}.html"


class RawStore:
    """HTML files in data/raw plus a manifest mapping each crawled file to its URL."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def save(self, url: str, html: str) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / _filename_for(url)
        path.write_text(html, encoding="utf-8")
        entry = {"url": url, "filename": path.name, "fetched_at": datetime.now(UTC).isoformat()}
        with (self._dir / MANIFEST_NAME).open("a", encoding="utf-8") as manifest:
            manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return path

    def _manifest_urls(self) -> dict[str, str]:
        manifest = self._dir / MANIFEST_NAME
        if not manifest.exists():
            return {}
        return {
            entry["filename"]: entry["url"]
            for entry in _read_jsonl(manifest)
            if "filename" in entry and "url" in entry
        }

    def urls(self) -> set[str]:
        """URLs already downloaded by the crawler and still present on disk."""
        return {
            url
            for filename, url in self._manifest_urls().items()
            if (self._dir / filename).exists()
        }

    def pages(self) -> Iterator[RawPage]:
        """Yield every HTML file in data/raw, crawled or saved by hand.

        Files that are not in the manifest get their URL from the HTML itself
        (canonical link, og:url or the browser's "saved from" comment). When
        none is present the page is kept under a local:// URL so its content
        is still searchable, and a warning is logged.
        """
        if not self._dir.exists():
            return
        known = self._manifest_urls()
        for path in sorted(self._dir.glob("*.htm*")):
            html = decode_html(path.read_bytes())
            url = known.get(path.name) or infer_url(html)
            if url is None:
                logger.warning(
                    "No source URL found for %s; using a local:// placeholder", path.name
                )
                url = f"local://{path.name}"
            yield RawPage(url=url, html=html, filename=path.name)


class CleanStore:
    """Clean documents as JSON Lines in data/clean/documents.jsonl."""

    def __init__(self, directory: Path) -> None:
        self._path = directory / DOCUMENTS_NAME

    @property
    def path(self) -> Path:
        return self._path

    def write(self, documents: Iterable[Document]) -> int:
        """Replace the file with `documents`; return how many were written.

        The new content is written to a temporary file and swapped in only when
        there is at least one document, so a failed or empty run never destroys
        the result of a previous good one.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".jsonl.tmp")
        count = 0
        try:
            with temporary.open("w", encoding="utf-8") as output:
                for document in documents:
                    output.write(document.model_dump_json() + "\n")
                    count += 1
            if count:
                os.replace(temporary, self._path)
        finally:
            temporary.unlink(missing_ok=True)
        return count

    def read(self) -> list[Document]:
        if not self._path.exists():
            raise ScrapingError(f"{self._path} does not exist; run the cleaning step first")
        return [Document.model_validate(entry) for entry in _read_jsonl(self._path)]

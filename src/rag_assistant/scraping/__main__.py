"""Command line entrypoint: python -m rag_assistant.scraping {crawl,clean,run}."""

import argparse
import logging
import sys

from rag_assistant.config import Settings, get_settings
from rag_assistant.exceptions import RAGError, ScrapingError
from rag_assistant.logging_config import configure_logging
from rag_assistant.scraping.cleaner import HtmlCleaner
from rag_assistant.scraping.crawler import SiteCrawler
from rag_assistant.scraping.storage import CleanStore, RawStore

logger = logging.getLogger(__name__)


def crawl(settings: Settings) -> int:
    """Download pages into data/raw; return how many were saved."""
    return SiteCrawler(settings, RawStore(settings.raw_data_dir)).crawl().saved


def clean(settings: Settings) -> int:
    """Clean everything in data/raw into data/clean; return the document count."""
    cleaner = HtmlCleaner(settings.scraper_min_text_chars)
    store = CleanStore(settings.clean_data_dir)
    count = store.write(cleaner.clean_all(RawStore(settings.raw_data_dir).pages()))
    if count == 0:
        raise ScrapingError(
            f"No usable pages in {settings.raw_data_dir}. Run the crawl step or save "
            "pages from your browser into that folder."
        )
    logger.info("Wrote %d clean documents to %s", count, store.path)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag_assistant.scraping")
    parser.add_argument(
        "command",
        choices=("crawl", "clean", "run"),
        help="crawl: download pages; clean: build the JSONL from data/raw; run: both",
    )
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
        configure_logging(settings.log_level)
        if args.command in ("crawl", "run"):
            try:
                crawl(settings)
            except ScrapingError as exc:
                if args.command == "crawl":
                    raise
                # "run" still cleans whatever is already in data/raw (e.g. pages saved by hand).
                logger.warning("Crawl failed, continuing with existing raw pages: %s", exc)
        if args.command in ("clean", "run"):
            clean(settings)
    except RAGError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

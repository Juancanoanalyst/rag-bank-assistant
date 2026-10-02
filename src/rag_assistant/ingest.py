"""One-shot ingestion: python -m rag_assistant.ingest [--force].

Runs crawl -> clean -> index, and does nothing when the vector store is already
populated. This is what the `ingest` service of docker-compose executes, so
`docker compose up` is fast after the first start.
"""

import argparse
import logging
import sys

from rag_assistant.config import get_settings
from rag_assistant.exceptions import RAGError, ScrapingError
from rag_assistant.factory import build_indexer
from rag_assistant.logging_config import configure_logging
from rag_assistant.scraping import __main__ as scraping
from rag_assistant.scraping.storage import CleanStore

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag_assistant.ingest")
    parser.add_argument("--force", action="store_true", help="re-crawl and re-index everything")
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
        configure_logging(settings.log_level)
        indexer = build_indexer(settings)
        if not args.force and indexer.is_indexed():
            logger.info("Content is already indexed; nothing to do (use --force to rebuild)")
            return 0

        try:
            scraping.crawl(settings)
        except ScrapingError as exc:
            # Pages saved by hand, or kept from an earlier run, are still usable.
            logger.warning("Crawl failed, continuing with existing raw pages: %s", exc)
        scraping.clean(settings)
        indexer.index(CleanStore(settings.clean_data_dir).read())
    except RAGError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

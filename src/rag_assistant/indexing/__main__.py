"""Command line entrypoint: python -m rag_assistant.indexing [--skip-if-indexed]."""

import argparse
import logging
import sys

from rag_assistant.config import get_settings
from rag_assistant.exceptions import RAGError
from rag_assistant.factory import build_indexer
from rag_assistant.logging_config import configure_logging
from rag_assistant.scraping.storage import CleanStore

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rag_assistant.indexing")
    parser.add_argument(
        "--skip-if-indexed",
        action="store_true",
        help="do nothing when the collection already contains chunks",
    )
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
        configure_logging(settings.log_level)
        documents = CleanStore(settings.clean_data_dir).read()
        build_indexer(settings).index(documents, skip_if_indexed=args.skip_if_indexed)
    except RAGError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

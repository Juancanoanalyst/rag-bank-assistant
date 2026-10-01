"""Logging setup shared by the API, the UI and the ingestion scripts."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"

# Third-party loggers that are too chatty at INFO.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3")


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once. Safe to call from every entrypoint."""
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

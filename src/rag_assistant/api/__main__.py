"""Run the API: python -m rag_assistant.api."""

import uvicorn

from rag_assistant.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "rag_assistant.api.app:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

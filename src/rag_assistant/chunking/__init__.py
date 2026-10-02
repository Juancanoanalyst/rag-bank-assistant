"""Ingestion stage 2: split clean documents into overlapping chunks."""

from rag_assistant.chunking.splitter import RecursiveTextSplitter

__all__ = ["RecursiveTextSplitter"]

"""Canonical ASGI entry point for ``hermes verify`` and Uvicorn."""

from docker.server import app

__all__ = ["app"]

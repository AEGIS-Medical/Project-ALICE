"""Compression pipeline package.

Public surface intentionally kept minimal -- callers should depend on the
orchestrator (``CompressionPipeline``) and the mode enum, not on the
individual stage classes. Stages can be imported directly from their
modules when needed (e.g. for tests or for building a custom orchestrator).
"""

from backend.shared.schemas.media import CompressionMode
from backend.workers.app.compression.pipeline import CompressionPipeline

__all__ = ["CompressionMode", "CompressionPipeline"]

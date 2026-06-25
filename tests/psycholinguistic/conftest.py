"""Test fixtures for the psycholinguistic analyzer suite.

The analyzer lives under ``backend/ml-inference/`` -- a service source root
whose directory name contains a hyphen, so it cannot be imported via a dotted
``backend.ml_inference`` path. Each ml-inference service runs with its own
directory on ``sys.path`` (so ``from app.pipelines... import ...`` resolves);
we replicate that here for the tests and CLI rather than renaming the spec's
mandated path. A single shared analyzer instance is provided because spaCy
model load (~0.5 s) dominates per-test time and the analyzer is stateless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ML_INFERENCE_ROOT = (
    Path(__file__).resolve().parents[2] / "backend" / "ml-inference"
)
if str(_ML_INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_INFERENCE_ROOT))


@pytest.fixture(scope="session")
def analyzer():
    """Session-scoped analyzer; spaCy is lazy-loaded on first use."""
    from app.pipelines.psycholinguistic.analyzer import PsycholinguisticAnalyzer

    return PsycholinguisticAnalyzer()

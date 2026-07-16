"""Language gate tests (gap #8 near-term): non-English never flows silently."""
from __future__ import annotations

import pytest

from app.pipelines.psycholinguistic.analyzer import (
    SUPPORTED_LANGUAGES,
    PsycholinguisticAnalyzer,
    UnsupportedLanguageError,
)

_STMTS = ["I think I was at home.", "I never went there."]


@pytest.fixture(scope="module")
def gate_analyzer():
    return PsycholinguisticAnalyzer()


def test_supported_languages_constant():
    assert SUPPORTED_LANGUAGES == frozenset({"en"})


@pytest.mark.parametrize("lang", ["en", "EN", "en-US", "en_GB", "en-au"])
def test_english_variants_accepted(gate_analyzer, lang):
    result = gate_analyzer.analyze(_STMTS, language=lang)
    assert result.statement_count == 2


@pytest.mark.parametrize("lang", ["es", "fr", "zh", "de", "ja"])
def test_non_english_raises(gate_analyzer, lang):
    with pytest.raises(UnsupportedLanguageError) as exc_info:
        gate_analyzer.analyze(_STMTS, language=lang)
    msg = str(exc_info.value)
    assert lang.lower() in msg
    assert "en" in msg  # names the supported set
    # Invariant #3: the error must not leak transcript text.
    assert "home" not in msg and "never went" not in msg


def test_none_language_preserves_legacy_behavior(gate_analyzer):
    result = gate_analyzer.analyze(_STMTS)  # no language kwarg at all
    assert result.statement_count == 2


def test_gate_precedes_empty_statement_check(gate_analyzer):
    """A Spanish EMPTY transcript reports the language error -- the more
    actionable fact -- not 'No statements provided'."""
    with pytest.raises(UnsupportedLanguageError):
        gate_analyzer.analyze([], language="es")


def test_empty_statements_still_raise_valueerror_when_english(gate_analyzer):
    with pytest.raises(ValueError, match="No statements provided"):
        gate_analyzer.analyze([], language="en")


def test_error_is_a_valueerror_subclass():
    assert issubclass(UnsupportedLanguageError, ValueError)


def test_package_reexports():
    from app.pipelines.psycholinguistic import (  # noqa: F401
        PsycholinguisticAnalyzer as A,
        SUPPORTED_LANGUAGES as S,
        UnsupportedLanguageError as E,
    )

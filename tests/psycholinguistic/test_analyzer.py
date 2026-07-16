"""Tests for the PsycholinguisticAnalyzer dimension scorers (P2-S2..S9).

Each scorer is unit-tested in isolation against hand-crafted statements whose
linguistic profile is unambiguous, plus the composite ``analyze`` entry point.
The shared session-scoped ``analyzer`` fixture (see conftest) lazy-loads spaCy
once. Scorers that take a parsed ``Doc`` are fed ``analyzer.nlp(text)``.
"""

from __future__ import annotations

import pytest

from backend.shared.schemas.psycholinguistic import (
    PsycholinguisticDimension,
    PsycholinguisticScore,
)


# ---- P2-S2: pronoun pattern scorer ----------------------------------------


def test_low_pronoun_density_scores_high(analyzer):
    # No first-person singular pronouns at all -> deceptive per Newman 2003.
    doc = analyzer.nlp(
        "The meeting occurred yesterday and the report was filed by the team."
    )
    dim = analyzer._score_pronouns(doc)
    assert isinstance(dim, PsycholinguisticDimension)
    assert dim.score > 50


def test_normal_pronoun_density_scores_low(analyzer):
    # One first-person singular token across ~17 tokens -> ratio ~0.06,
    # squarely in the normal 0.03-0.12 band.
    doc = analyzer.nlp(
        "I walked down to the old market near the river to buy fresh bread."
    )
    dim = analyzer._score_pronouns(doc)
    assert dim.score < 50


def test_evidence_populated(analyzer):
    doc = analyzer.nlp("I think I saw my friend.")
    dim = analyzer._score_pronouns(doc)
    assert dim.evidence  # non-empty


# ---- P2-S3: hedging scorer ------------------------------------------------


def test_high_hedging_statement_scores_high(analyzer):
    doc = analyzer.nlp("I think maybe it could perhaps have been the case.")
    dim = analyzer._score_hedging(doc)
    assert dim.score > 60


def test_direct_statement_scores_low(analyzer):
    doc = analyzer.nlp("I did it at 3pm.")
    dim = analyzer._score_hedging(doc)
    assert dim.score < 30


def test_hedge_evidence_populated(analyzer):
    doc = analyzer.nlp("I think maybe it could have happened.")
    dim = analyzer._score_hedging(doc)
    assert dim.evidence  # at least one detected hedge surfaced


# ---- P2-S4: cognitive complexity scorer -----------------------------------


def test_complex_sentence_scores_low(analyzer):
    doc = analyzer.nlp(
        "I believe that the man who arrived knew that she had left "
        "because the door was open."
    )
    dim = analyzer._score_cognitive_complexity(doc)
    assert dim.score < 40


def test_simple_sentences_score_high(analyzer):
    doc = analyzer.nlp("The cat sat. The dog ran. The bird flew.")
    dim = analyzer._score_cognitive_complexity(doc)
    assert dim.score > 60


def test_cognitive_complexity_multi_sentence(analyzer):
    doc = analyzer.nlp(
        "I went to the store because I was hungry. I bought bread that was fresh."
    )
    dim = analyzer._score_cognitive_complexity(doc)
    assert isinstance(dim, PsycholinguisticDimension)
    assert 0.0 <= dim.score <= 100.0


# ---- P2-S5: emotional word distribution scorer ----------------------------


def test_anxiety_anger_text_scores_high(analyzer):
    dim = analyzer._score_emotional_distribution(
        "I am terrified and furious and scared and angry and anxious about this threat."
    )
    assert dim.score > 60


def test_positive_family_text_scores_low(analyzer):
    dim = analyzer._score_emotional_distribution(
        "My loving family enjoyed a wonderful happy joyful celebration together at home."
    )
    assert dim.score < 40


# ---- P2-S6: disfluency scorer ---------------------------------------------


def test_disfluent_text_scores_high(analyzer):
    dim = analyzer._score_disfluencies("Um, uh, so um I uh went there um er.")
    assert dim.score > 50
    assert dim.evidence  # detected disfluencies listed


def test_clean_text_scores_low(analyzer):
    dim = analyzer._score_disfluencies(
        "I drove to the office and finished the quarterly report on time."
    )
    assert dim.score < 20


# ---- P2-S7: negation scorer -----------------------------------------------


def test_high_negation_scores_high(analyzer):
    doc = analyzer.nlp("I never did, I did not, I was not there.")
    dim = analyzer._score_negation(doc)
    assert dim.score > 60


def test_affirmative_scores_low(analyzer):
    doc = analyzer.nlp("I completed the task and signed the form.")
    dim = analyzer._score_negation(doc)
    assert dim.score < 30


# ---- P2-S7: detail specificity scorer -------------------------------------


def test_entity_rich_scores_low(analyzer):
    doc = analyzer.nlp(
        "On Monday John drove to Chicago to meet Sarah at noon near the river."
    )
    dim = analyzer._score_detail_specificity(doc)
    assert dim.score < 40


def test_vague_text_scores_high(analyzer):
    doc = analyzer.nlp("I went somewhere and did some stuff with someone at some point.")
    dim = analyzer._score_detail_specificity(doc)
    assert dim.score > 60


# ---- P2-S8: certainty/tentative language scorer ---------------------------


def test_over_certain_scores_high(analyzer):
    text = "I absolutely definitely 100% did not do it, no doubt whatsoever."
    doc = analyzer.nlp(text)
    dim = analyzer._score_certainty(doc, text)
    assert dim.score > 60


def test_neutral_confident_scores_low(analyzer):
    text = "I finished the report and sent it to the team."
    doc = analyzer.nlp(text)
    dim = analyzer._score_certainty(doc, text)
    assert dim.score < 40


# ---- P2-S9: composite analyze() -------------------------------------------

_THREE_STATEMENTS = [
    "I went to the store on Monday and bought groceries.",
    "Maybe I think it could have happened, I'm not sure.",
    "I never did that, I was not there at all.",
]

_DIMENSION_FIELDS = (
    "pronoun_shift_score",
    "hedging_score",
    "cognitive_complexity_score",
    "emotional_distribution_score",
    "disfluency_score",
    "negation_score",
    "detail_specificity_score",
    "certainty_score",
)


def test_analyze_empty_raises(analyzer):
    with pytest.raises(ValueError, match="No statements provided"):
        analyzer.analyze([])


def test_analyze_returns_all_dimensions(analyzer):
    result = analyzer.analyze(_THREE_STATEMENTS)
    assert isinstance(result, PsycholinguisticScore)
    for field in _DIMENSION_FIELDS:
        dim = getattr(result, field)
        assert isinstance(dim, PsycholinguisticDimension)
    assert result.statement_count == 3
    assert isinstance(result.composite_score, float)


def test_composite_score_in_range(analyzer):
    result = analyzer.analyze(_THREE_STATEMENTS)
    assert 0.0 <= result.composite_score <= 100.0


def test_analyze_deterministic(analyzer):
    a = analyzer.analyze(_THREE_STATEMENTS)
    b = analyzer.analyze(_THREE_STATEMENTS)
    assert a.composite_score == b.composite_score
    for field in _DIMENSION_FIELDS:
        assert getattr(a, field).score == getattr(b, field).score


# ---- Session 6 / gap #4: hedging-certainty disjoint ------------------------


def test_hedge_and_certainty_lists_are_disjoint():
    """Regression guard for gap #4: no marker may appear in both lists."""
    from app.pipelines.psycholinguistic import analyzer as mod

    overlap = set(mod._HEDGE_PHRASES) & set(mod._CERTAINTY_MARKERS)
    assert not overlap, f"markers scored twice: {sorted(overlap)}"


def test_tentative_markers_list_is_gone():
    """The tentative list was deleted; a revert must be loud."""
    from app.pipelines.psycholinguistic import analyzer as mod

    assert not hasattr(mod, "_TENTATIVE_MARKERS")


def test_tentative_text_no_longer_raises_certainty(analyzer):
    """Tentative-only text is hedging's job now: certainty stays low while
    hedging stays high (each signal counted exactly once)."""
    text = "I think maybe it was sort of possibly like that, I guess."
    doc = analyzer.nlp(text)
    certainty = analyzer._score_certainty(doc, text)
    hedging = analyzer._score_hedging(doc)
    assert certainty.score <= 35.0, (
        f"tentative text should barely move over-certainty "
        f"(got {certainty.score})"
    )
    assert hedging.score > 50.0, f"hedging must still fire (got {hedging.score})"


def test_emphatic_text_still_maxes_certainty(analyzer):
    """The over-certainty signal is untouched by the disjoint."""
    text = "I absolutely definitely swear, 100%, I never ever did it, no doubt."
    doc = analyzer.nlp(text)
    certainty = analyzer._score_certainty(doc, text)
    assert certainty.score >= 60.0, f"emphatic text must score high (got {certainty.score})"

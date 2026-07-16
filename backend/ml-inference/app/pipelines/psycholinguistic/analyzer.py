"""Psycholinguistic analyzer for Project ALICE (Phase 2, Day 1).

Scores speaker-attributed statement text across the eight deception-relevant
linguistic dimensions defined in CLAUDE.md, producing a
``PsycholinguisticScore``. Linguistic features are the strongest single
modality (~80% accuracy; Li & Abouelenien 2024) and this vector carries 0.30
of the late-fusion ensemble weight.

This is a *behavioral anomaly* scorer, not a verdict engine: every score is a
deviation signal, never a probability of deception (CLAUDE.md). Day 1 uses
lightweight, fully-local tooling -- spaCy ``en_core_web_sm``, NRCLex, and VADER
-- with no large model downloads. Several heuristics (notably hedging) are
explicitly marked for replacement with fine-tuned classifiers in Phase 3.

Tooling (CLAUDE.md "Psycholinguistic Analysis Stack"):
    spaCy en_core_web_sm   POS, dependency parse, NER, pronouns, negation
    NRCLex                 8 granular emotion categories
    VADER                  valence-aware sentiment / certainty
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from backend.shared.schemas.psycholinguistic import (
    PsycholinguisticDimension,
    PsycholinguisticScore,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spacy.language import Language
    from spacy.tokens import Doc

logger = logging.getLogger(__name__)

# spaCy model used for all parsing. Small (~12 MB), no GPU, deterministic.
_SPACY_MODEL = "en_core_web_sm"

# First-person singular tokens (lower-cased lemma/text match) per Newman et al.
# (2003): a *drop* in first-person singular usage is deception-indicative.
_FIRST_PERSON_SINGULAR = frozenset({"i", "me", "my", "mine", "myself"})

# Pronoun ratio bands. Inside [LOW, HIGH] is the normal, low-anomaly range;
# below LOW (suppression) or above HIGH (over-compensation) is anomalous.
_PRONOUN_RATIO_LOW = 0.03
_PRONOUN_RATIO_HIGH = 0.12

# Curated epistemic hedge phrases (Day 1). NOTE: this word-list approach has a
# ~59% false-positive rate (CLAUDE.md) and MUST be replaced by a fine-tuned
# BERT hedging classifier in Phase 3. Matched case-insensitively as substrings.
_HEDGE_PHRASES = (
    "i think",
    "i believe",
    "i guess",
    "i suppose",
    "i feel like",
    "i'm not sure",
    "im not sure",
    "not sure",
    "sort of",
    "kind of",
    "maybe",
    "perhaps",
    "possibly",
    "probably",
    "or something",
    "you know",
)

# Each hedge contributes this many points; the per-sentence hedge rate is
# multiplied by it and capped at 100.
_HEDGE_POINTS_PER_RATE = 20.0

# Subordinate-clause dependency arcs that mark cognitive complexity. Truth-
# tellers tend to use more complex, layered syntax; deceivers keep statements
# simple to stay consistent -- so higher complexity LOWERS the anomaly score.
_SUBORDINATE_DEPS = frozenset({"advcl", "relcl", "ccomp", "xcomp"})

# Points subtracted from 100 per subordinate clause per sentence.
_COMPLEXITY_POINTS_PER_RATE = 25.0

# NRCLex emotion categories grouped by deception direction (Pérez-Rosas, EMNLP
# 2015): deceivers skew toward anxiety/anger/fear; truth-tellers toward
# positive/trust/family words. NRCLex lacks a distinct "anxiety" axis, so fear
# + negative stand in for it.
_DECEPTIVE_EMOTIONS = ("fear", "anger", "negative")
_TRUTHFUL_EMOTIONS = ("positive", "trust", "joy")

# Simple word tokenizer for NRCLex (avoids its TextBlob/nltk corpus dependency).
_WORD_RE = re.compile(r"[a-z']+")

# Single-token filler words and multi-word filler phrases (cognitive-load
# signal; higher disfluency RAISES the anomaly score). "like" is included as a
# Day 1 heuristic despite false positives on its verb/preposition senses.
_FILLER_WORDS = frozenset({"um", "uh", "er", "ah", "like"})
_FILLER_PHRASES = ("you know",)

# Disfluency score = (fillers / words) scaled by this. Equivalent to "percent
# of words that are fillers" at 100.
_DISFLUENCY_POINTS_PER_RATE = 100.0

# Negation arcs per sentence scaled by this (more negation -> higher anomaly).
_NEGATION_POINTS_PER_RATE = 25.0

# Named-entity density subtracted from 100 (less specific -> higher anomaly).
# NE density rarely exceeds ~0.35, so the multiplier is large.
_SPECIFICITY_POINTS_PER_DENSITY = 200.0

# Over-certainty / emphatic-assertion markers ("protesting too much").
# Session 6 / gap #4: tentative markers were REMOVED from this dimension --
# every one of them also appeared in _HEDGE_PHRASES, double-counting
# tentative language across two of the eight equally-weighted dimensions.
# Tentative language is owned solely by the hedging dimension; this
# dimension fires on emphatic absolutes only. Matched case-insensitively
# as substrings.
_CERTAINTY_MARKERS = (
    "definitely",
    "absolutely",
    "certainly",
    "undoubtedly",
    "no doubt",
    "without a doubt",
    "100%",
    "for sure",
    "i guarantee",
    "i swear",
    "totally",
    "completely",
    "always",
    "never",
)

# Points per detected extremity marker, plus a VADER-intensity contribution.
_CERTAINTY_POINTS_PER_MARKER = 25.0
_CERTAINTY_VADER_WEIGHT = 30.0


class PsycholinguisticAnalyzer:
    """Scores statement text across eight psycholinguistic dimensions.

    spaCy is lazy-loaded on first parse so constructing the analyzer is cheap
    and import-safe in environments where the model is not yet downloaded. The
    analyzer is stateless and deterministic: the same input always yields the
    same scores, which the late-fusion ensemble and tests rely on.
    """

    def __init__(self) -> None:
        self._nlp: Language | None = None
        self._vader = None

    # ---- public API --------------------------------------------------------

    def analyze(self, statements: list[str]) -> PsycholinguisticScore:
        """Score a list of statements across all eight dimensions.

        Statements are concatenated into a single document for parsing (Day 1;
        per-statement and per-contact baselining arrive in later phases). The
        composite is the equally-weighted mean of the eight dimension scores --
        the linguistic vector's internal sub-weights are equal for Day 1
        (CLAUDE.md; the 0.30 ensemble weight is applied later at fusion time).

        Args:
            statements: Speaker-attributed statement strings.

        Returns:
            A frozen ``PsycholinguisticScore``.

        Raises:
            ValueError: ``statements`` is empty.
        """
        if not statements:
            raise ValueError("No statements provided")

        text = " ".join(statements)
        doc = self.nlp(text)

        dimensions = {
            "pronoun_shift_score": self._score_pronouns(doc),
            "hedging_score": self._score_hedging(doc),
            "cognitive_complexity_score": self._score_cognitive_complexity(doc),
            "emotional_distribution_score": self._score_emotional_distribution(text),
            "disfluency_score": self._score_disfluencies(text),
            "negation_score": self._score_negation(doc),
            "detail_specificity_score": self._score_detail_specificity(doc),
            "certainty_score": self._score_certainty(doc, text),
        }

        composite = sum(dim.score for dim in dimensions.values()) / len(dimensions)

        # Day 1: no per-contact history is wired up yet, so a single analysis
        # pass cannot establish a baseline. Report this honestly rather than
        # implying confidence we do not have (CLAUDE.md invariants).
        return PsycholinguisticScore(
            **dimensions,
            composite_score=composite,
            statement_count=len(statements),
            baseline_available=False,
            confidence="low",
        )

    # ---- spaCy lazy loader -------------------------------------------------

    @property
    def nlp(self) -> Language:
        """Return the loaded spaCy pipeline, loading it on first access."""
        if self._nlp is None:
            import spacy

            try:
                self._nlp = spacy.load(_SPACY_MODEL)
            except OSError as exc:  # model not installed
                raise RuntimeError(
                    f"spaCy model {_SPACY_MODEL!r} is not installed. "
                    f"Run: python -m spacy download {_SPACY_MODEL}"
                ) from exc
        return self._nlp

    @property
    def vader(self):
        """Return the VADER sentiment analyzer, constructing it on first use."""
        if self._vader is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._vader = SentimentIntensityAnalyzer()
        return self._vader

    # ---- P2-S2: pronoun pattern scorer ------------------------------------

    def _score_pronouns(self, doc: Doc) -> PsycholinguisticDimension:
        """Score first-person-singular usage (Newman et al. 2003).

        Computes the first-person-singular ratio over all tokens. A ratio below
        ``_PRONOUN_RATIO_LOW`` (suppression) or above ``_PRONOUN_RATIO_HIGH``
        (over-compensation; motivated liars may *increase* self-reference)
        scores high; the normal midband scores low.
        """
        total = sum(1 for tok in doc if not tok.is_space)
        fps_count = sum(
            1 for tok in doc if tok.text.lower() in _FIRST_PERSON_SINGULAR
        )
        ratio = (fps_count / total) if total else 0.0
        score = self._pronoun_ratio_to_score(ratio)
        evidence = [
            f"first_person_singular_ratio={ratio:.3f}",
            f"first_person_singular_count={fps_count}",
            f"token_count={total}",
        ]
        return PsycholinguisticDimension(score=score, evidence=evidence)

    @staticmethod
    def _pronoun_ratio_to_score(ratio: float) -> float:
        """Map a first-person-singular ratio to a 0-100 anomaly score.

        Inside the normal band the score sits low (20); outside it ramps from
        the band edge up to 100 at the extremes (ratio 0, or >= 2x the high
        edge).
        """
        if _PRONOUN_RATIO_LOW <= ratio <= _PRONOUN_RATIO_HIGH:
            return 20.0
        if ratio < _PRONOUN_RATIO_LOW:
            # 0 -> 100, band edge -> 60.
            frac = (_PRONOUN_RATIO_LOW - ratio) / _PRONOUN_RATIO_LOW
            return min(100.0, 60.0 + 40.0 * frac)
        # ratio > HIGH: band edge -> 60, 2x edge -> 100.
        frac = (ratio - _PRONOUN_RATIO_HIGH) / _PRONOUN_RATIO_HIGH
        return min(100.0, 60.0 + 40.0 * frac)

    # ---- P2-S3: hedging scorer --------------------------------------------

    def _score_hedging(self, doc: Doc) -> PsycholinguisticDimension:
        """Score hedging via modal verbs + curated epistemic phrases.

        Modal verbs are detected by spaCy's ``MD`` fine-grained POS tag;
        epistemic phrases by case-insensitive substring match. The combined
        hedge count is normalized per sentence and scaled to 0-100.

        Day 1 heuristic only -- replace with a fine-tuned BERT hedging
        classifier in Phase 3 (CLAUDE.md: the word-list has a 59% FP rate).
        """
        modals = [tok.text.lower() for tok in doc if tok.tag_ == "MD"]
        text_lower = doc.text.lower()
        phrase_hits: list[str] = []
        for phrase in _HEDGE_PHRASES:
            count = text_lower.count(phrase)
            phrase_hits.extend([phrase] * count)

        hedges = modals + phrase_hits
        n_sentences = max(1, sum(1 for _ in doc.sents))
        rate = len(hedges) / n_sentences
        score = min(100.0, rate * _HEDGE_POINTS_PER_RATE)

        # Evidence: the three most frequent detected hedges.
        top = sorted(set(hedges), key=lambda h: (-hedges.count(h), h))[:3]
        evidence = [f"hedge_count={len(hedges)}", f"hedges_per_sentence={rate:.2f}"]
        if top:
            evidence.append("top_hedges=" + ", ".join(top))
        return PsycholinguisticDimension(score=score, evidence=evidence)

    # ---- P2-S4: cognitive complexity scorer -------------------------------

    def _score_cognitive_complexity(self, doc: Doc) -> PsycholinguisticDimension:
        """Score subordinate-clause depth (inverse anomaly).

        Counts ``advcl``/``relcl``/``ccomp``/``xcomp`` dependency arcs,
        normalizes per sentence, and inverts: richer syntax (truth-indicative)
        yields a lower anomaly score; flat simple syntax yields a higher one.
        """
        arcs = [tok.dep_ for tok in doc if tok.dep_ in _SUBORDINATE_DEPS]
        n_sentences = max(1, sum(1 for _ in doc.sents))
        rate = len(arcs) / n_sentences
        score = max(0.0, 100.0 - rate * _COMPLEXITY_POINTS_PER_RATE)
        evidence = [
            f"subordinate_clauses={len(arcs)}",
            f"sentences={n_sentences}",
            f"clauses_per_sentence={rate:.2f}",
        ]
        return PsycholinguisticDimension(score=score, evidence=evidence)

    # ---- P2-S5: emotional word distribution scorer ------------------------

    def _score_emotional_distribution(self, text: str) -> PsycholinguisticDimension:
        """Score emotional-word balance via NRCLex (Pérez-Rosas 2015).

        Deception-indicative emotions (fear/anger/negative) push the score up;
        truth-indicative emotions (positive/trust/joy) push it down. The net
        frequency difference, in roughly [-1, 1], maps linearly onto 0-100 with
        a neutral 50 midpoint.

        Tokenizes with a simple regex and feeds NRCLex via ``load_token_list``
        to avoid its optional TextBlob/nltk corpus dependency.
        """
        from nrclex import NRCLex

        tokens = _WORD_RE.findall(text.lower())
        nrc = NRCLex()
        nrc.load_token_list(tokens)
        freqs = nrc.affect_frequencies

        deceptive = sum(freqs.get(e, 0.0) for e in _DECEPTIVE_EMOTIONS)
        truthful = sum(freqs.get(e, 0.0) for e in _TRUTHFUL_EMOTIONS)
        net = deceptive - truthful
        score = max(0.0, min(100.0, 50.0 + net * 50.0))
        evidence = [
            f"deceptive_emotion_freq={deceptive:.3f}",
            f"truthful_emotion_freq={truthful:.3f}",
            f"net={net:.3f}",
        ]
        return PsycholinguisticDimension(score=score, evidence=evidence)

    # ---- P2-S6: disfluency scorer -----------------------------------------

    def _score_disfluencies(self, text: str) -> PsycholinguisticDimension:
        """Score filler-word frequency (cognitive-load signal).

        Counts single-token fillers (um/uh/er/ah/like) and multi-word filler
        phrases (you know), normalized by total word count. More disfluency
        RAISES the anomaly score.
        """
        words = _WORD_RE.findall(text.lower())
        word_count = len(words)
        text_lower = text.lower()

        detected: dict[str, int] = {}
        for w in words:
            if w in _FILLER_WORDS:
                detected[w] = detected.get(w, 0) + 1
        for phrase in _FILLER_PHRASES:
            count = text_lower.count(phrase)
            if count:
                detected[phrase] = detected.get(phrase, 0) + count

        filler_count = sum(detected.values())
        rate = (filler_count / word_count) if word_count else 0.0
        score = min(100.0, rate * _DISFLUENCY_POINTS_PER_RATE)
        evidence = [f"filler_count={filler_count}", f"word_count={word_count}"]
        if detected:
            top = sorted(detected, key=lambda k: (-detected[k], k))
            evidence.append(
                "detected=" + ", ".join(f"{k}({detected[k]})" for k in top)
            )
        return PsycholinguisticDimension(score=score, evidence=evidence)

    # ---- P2-S7: negation scorer -------------------------------------------

    def _score_negation(self, doc: Doc) -> PsycholinguisticDimension:
        """Score negation frequency (``neg`` arcs per sentence).

        Frequent negation is associated with deceptive denial patterns, so a
        higher per-sentence negation rate RAISES the anomaly score.
        """
        neg_arcs = [tok.text.lower() for tok in doc if tok.dep_ == "neg"]
        n_sentences = max(1, sum(1 for _ in doc.sents))
        rate = len(neg_arcs) / n_sentences
        score = min(100.0, rate * _NEGATION_POINTS_PER_RATE)
        evidence = [
            f"negation_count={len(neg_arcs)}",
            f"sentences={n_sentences}",
            f"negations_per_sentence={rate:.2f}",
        ]
        return PsycholinguisticDimension(score=score, evidence=evidence)

    # ---- P2-S7: detail specificity scorer ---------------------------------

    def _score_detail_specificity(self, doc: Doc) -> PsycholinguisticDimension:
        """Score named-entity density (inverse anomaly).

        Specific accounts cite dates, places, and people; vague accounts do
        not. Named-entity density is inverted: low density (vague) yields a
        HIGHER anomaly score.
        """
        n_words = sum(1 for tok in doc if not tok.is_space and not tok.is_punct)
        n_entities = len(doc.ents)
        density = (n_entities / n_words) if n_words else 0.0
        score = max(0.0, 100.0 - density * _SPECIFICITY_POINTS_PER_DENSITY)
        labels = sorted({ent.label_ for ent in doc.ents})
        evidence = [
            f"named_entities={n_entities}",
            f"word_count={n_words}",
            f"entity_density={density:.3f}",
        ]
        if labels:
            evidence.append("entity_types=" + ", ".join(labels))
        return PsycholinguisticDimension(score=score, evidence=evidence)

    # ---- P2-S8: over-certainty / emphatic assertion scorer -----------------

    def _score_certainty(self, doc: Doc, text: str) -> PsycholinguisticDimension:
        """Score over-certainty / emphatic assertion (dimension 8).

        Counts absolute-certainty markers ("definitely", "i swear", "100%")
        and folds in VADER's sentiment intensity (|compound|). Tentative
        language is deliberately NOT counted here -- it is the hedging
        dimension's signal (gap #4 disjoint; each marker counted once).
        ``doc`` is accepted for interface symmetry with the other scorers.
        """
        text_lower = text.lower()
        certainty_hits = [m for m in _CERTAINTY_MARKERS if m in text_lower]
        marker_count = sum(text_lower.count(m) for m in certainty_hits)

        compound = self.vader.polarity_scores(text)["compound"]
        intensity = abs(compound)

        score = min(
            100.0,
            marker_count * _CERTAINTY_POINTS_PER_MARKER
            + intensity * _CERTAINTY_VADER_WEIGHT,
        )
        evidence = [
            f"overcertainty_markers={marker_count}",
            f"vader_compound={compound:.3f}",
        ]
        if certainty_hits:
            evidence.append("markers=" + ", ".join(sorted(set(certainty_hits))))
        return PsycholinguisticDimension(score=score, evidence=evidence)

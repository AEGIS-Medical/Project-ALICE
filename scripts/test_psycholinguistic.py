#!/usr/bin/env python
"""CLI smoke test for PsycholinguisticAnalyzer.

The analyzer lives under ``backend/ml-inference/`` (a hyphenated service root
that cannot be imported via a dotted path), so we insert that root onto
``sys.path`` exactly as ``tests/psycholinguistic/conftest.py`` does, then import
``from app.pipelines...``.

Usage:
    python scripts/test_psycholinguistic.py --text "I think maybe I never did that."
    python scripts/test_psycholinguistic.py --file path/to/statements.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ML_INFERENCE_ROOT = _REPO_ROOT / "backend" / "ml-inference"
if str(_ML_INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_INFERENCE_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the psycholinguistic analyzer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text string to analyze")
    group.add_argument("--file", type=Path, help="Path to a text file (one statement per line)")
    args = parser.parse_args()

    if args.text:
        statements = [args.text]
    else:
        if not args.file.exists():
            print(f"ERROR: file not found: {args.file}", file=sys.stderr)
            return 1
        statements = [
            line.strip()
            for line in args.file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not statements:
        print("ERROR: no statements to analyze.", file=sys.stderr)
        return 1

    from app.pipelines.psycholinguistic.analyzer import PsycholinguisticAnalyzer

    analyzer = PsycholinguisticAnalyzer()
    result = analyzer.analyze(statements)

    # (label, dimension) pairs in CLAUDE.md order, using the real field names.
    dimensions = [
        ("Pronoun Shift", result.pronoun_shift_score),
        ("Hedging", result.hedging_score),
        ("Cognitive Complexity", result.cognitive_complexity_score),
        ("Emotional Distribution", result.emotional_distribution_score),
        ("Disfluency", result.disfluency_score),
        ("Negation", result.negation_score),
        ("Detail Specificity", result.detail_specificity_score),
        ("Certainty", result.certainty_score),
    ]

    print(f"\nPsycholinguistic Analysis -- {result.statement_count} statement(s)")
    print(
        f"Confidence: {result.confidence} | "
        f"Baseline: {'available' if result.baseline_available else 'not yet established'}"
    )
    print("-" * 64)
    for name, dim in dimensions:
        bar = "#" * int(dim.score / 5)
        print(f"  {name:<24} {dim.score:6.1f}/100  {bar}")
        if dim.evidence:
            print(f"      {', '.join(dim.evidence[:3])}")
    print("-" * 64)
    bar = "#" * int(result.composite_score / 5)
    print(f"  {'COMPOSITE':<24} {result.composite_score:6.1f}/100  {bar}")
    print()
    print("NOTE: scores are behavioral anomaly signals, not ground truth.")
    print("      ~75% F1 realistic ceiling. ALICE is not a verdict engine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

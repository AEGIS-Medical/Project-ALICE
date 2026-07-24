from __future__ import annotations

from app.service.publisher import DROPPED, InProcessPublisher
from backend.shared.schemas.psycholinguistic import (
    PsycholinguisticDimension,
    PsycholinguisticScore,
)
from backend.shared.schemas.score_event import ScoreEvent, ScoreEventKind


def _dim(score: float = 10.0) -> PsycholinguisticDimension:
    return PsycholinguisticDimension(score=score, evidence=[])


def _event(t: float, kind=ScoreEventKind.INTERIM) -> ScoreEvent:
    score = PsycholinguisticScore(
        pronoun_shift_score=_dim(), hedging_score=_dim(),
        cognitive_complexity_score=_dim(), emotional_distribution_score=_dim(),
        disfluency_score=_dim(), negation_score=_dim(),
        detail_specificity_score=_dim(), certainty_score=_dim(),
        composite_score=10.0, statement_count=1,
        baseline_available=False, confidence="low",
    )
    return ScoreEvent(
        kind=kind, stream_time_seconds=t, cumulative=score, recent=None,
        vector_scores={"psycholinguistic": 10.0},
        statement_count_so_far=1, baseline_available=False, confidence="low",
    )


def test_seq_starts_at_zero_and_increments():
    pub = InProcessPublisher("s1", ring_size=10, queue_size=8)
    assert pub.last_seq == -1
    pub.publish(_event(5.0))
    pub.publish(_event(10.0))
    assert pub.last_seq == 1
    assert [f["seq"] for f in pub.buffered] == [0, 1]
    assert pub.buffered[0]["session_id"] == "s1"
    assert pub.buffered[0]["event"]["stream_time_seconds"] == 5.0


def test_ring_trims_to_ring_size():
    pub = InProcessPublisher("s1", ring_size=3, queue_size=8)
    for i in range(6):
        pub.publish(_event(float(i)))
    seqs = [f["seq"] for f in pub.buffered]
    assert seqs == [3, 4, 5]


def test_subscribe_catch_up_and_predates_window():
    pub = InProcessPublisher("s1", ring_size=3, queue_size=8)
    for i in range(6):
        pub.publish(_event(float(i)))
    q = pub.subscribe(last_seq=1)          # 1 predates window (oldest is 3)
    got = [q.get_nowait()["seq"] for _ in range(q.qsize())]
    assert got == [3, 4, 5]                # silent start at oldest buffered


def test_two_subscribers_receive_identically():
    pub = InProcessPublisher("s1", ring_size=10, queue_size=8)
    q1, q2 = pub.subscribe(), pub.subscribe()
    assert pub.subscriber_count == 2
    pub.publish(_event(5.0))
    assert q1.get_nowait() == q2.get_nowait()


def test_terminal_frame_shape_and_always_included():
    pub = InProcessPublisher("s1", ring_size=10, queue_size=8)
    pub.publish(_event(5.0))
    pub.publish_terminal("finished")
    frames = pub.buffered
    assert frames[-1] == {"session_id": "s1", "state": "finished", "reason": None}
    assert "seq" not in frames[-1]
    # A late subscriber that has seen everything still gets the terminal frame.
    q = pub.subscribe(last_seq=0)
    got = [q.get_nowait() for _ in range(q.qsize())]
    assert got[-1]["state"] == "finished"


def test_slow_subscriber_dropped_with_marker_session_unaffected():
    pub = InProcessPublisher("s1", ring_size=64, queue_size=2)
    slow = pub.subscribe()
    fine = pub.subscribe()
    for i in range(4):                     # overflows queue_size=2
        pub.publish(_event(float(i)))
        while fine.qsize():
            fine.get_nowait()
    assert pub.subscriber_count == 1       # only slow removed
    drained = [slow.get_nowait() for _ in range(slow.qsize())]
    assert drained[-1] is DROPPED
    pub.publish(_event(99.0))
    assert pub.last_seq == 4               # publishing never stalled


def test_publish_after_terminal_is_ignored():
    """Terminal-is-always-last ring invariant: a late data frame from an
    in-flight worker callback (DELETE race) must not append past the terminal."""
    pub = InProcessPublisher("s1", ring_size=16, queue_size=8)
    pub.publish(_event(1.0))
    pub.publish_terminal("cancelled")
    pub.publish(_event(2.0))               # late frame from the racing worker
    assert pub.buffered[-1].get("state") == "cancelled"   # terminal still last
    assert pub.last_seq == 0               # seq did not advance past the terminal


def test_unsubscribe_idempotent():
    pub = InProcessPublisher("s1", ring_size=4, queue_size=4)
    q = pub.subscribe()
    pub.unsubscribe(q)
    pub.unsubscribe(q)
    assert pub.subscriber_count == 0


def test_late_subscriber_with_large_backlog_still_gets_terminal_frame():
    """Regression: catch-up larger than the queue must never squeeze out the
    terminal frame -- a reconnecting client would hang forever otherwise."""
    pub = InProcessPublisher("s1", ring_size=32, queue_size=4)
    for i in range(10):
        pub.publish(_event(float(i)))
    pub.publish_terminal("finished")

    q = pub.subscribe(last_seq=-1)      # backlog (11 frames) > queue_size (4)
    got = [q.get_nowait() for _ in range(q.qsize())]
    assert got[-1].get("state") == "finished"          # terminal delivered
    data_seqs = [f["seq"] for f in got if "seq" in f]
    assert data_seqs == sorted(data_seqs)               # newest tail, in order
    assert len(got) <= 4

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.service.config import LiveServiceConfig


def test_defaults():
    c = LiveServiceConfig()
    assert c.host == "127.0.0.1"
    assert c.port == 8710
    assert c.ring_size == 256
    assert c.subscriber_queue_size == 64
    assert c.session_ttl_seconds == 900.0
    assert c.reaper_interval_seconds == 5.0


@pytest.mark.parametrize(
    "field,value",
    [("ring_size", 0), ("subscriber_queue_size", 0),
     ("session_ttl_seconds", 0.0), ("reaper_interval_seconds", 0.0)],
)
def test_bounds(field, value):
    with pytest.raises(ValidationError):
        LiveServiceConfig(**{field: value})


def test_frozen():
    c = LiveServiceConfig()
    with pytest.raises(ValidationError):
        c.port = 9999

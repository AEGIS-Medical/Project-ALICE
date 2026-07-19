"""Live-service configuration (frozen, launcher-flag overridable)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LiveServiceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = Field(default="127.0.0.1", description="Bind host (localhost = v1 auth posture).")
    port: int = Field(default=8710, ge=1, le=65535)
    ring_size: int = Field(default=256, ge=1, description="Per-session event ring buffer length.")
    subscriber_queue_size: int = Field(default=64, ge=1, description="Per-connection queue; overflow drops that connection (4408).")
    session_ttl_seconds: float = Field(default=900.0, gt=0.0, description="Reap terminal (or stuck-CREATED) sessions after this long.")
    reaper_interval_seconds: float = Field(default=5.0, gt=0.0)

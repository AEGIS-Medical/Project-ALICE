#!/usr/bin/env python
"""Launch the ALICE live service (dev posture: localhost, no auth).

Usage:
    python scripts/run_live_service.py
    python scripts/run_live_service.py --port 9000 --host 0.0.0.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ML_INFERENCE_ROOT = _REPO_ROOT / "backend" / "ml-inference"
for _p in (_ML_INFERENCE_ROOT, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    parser = argparse.ArgumentParser(description="ALICE live service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8710)
    args = parser.parse_args()

    import uvicorn

    from app.service.app import create_app
    from app.service.config import LiveServiceConfig

    config = LiveServiceConfig(host=args.host, port=args.port)
    print(f"ALICE live service on http://{config.host}:{config.port}")
    print("Create a session:  POST /sessions   | Watch: WS /sessions/{id}/events")
    print("NOTE: dev surface -- events are ensemble/developer-facing only.")
    uvicorn.run(create_app(config), host=config.host, port=config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

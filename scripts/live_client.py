#!/usr/bin/env python
"""Minimal demo client: create a session and watch its event stream.

Usage:
    python scripts/live_client.py --transcript path/to/t.json --pace 1
    python scripts/live_client.py --video demo_data/honest/trial_truth_001.mp4 --fake
    python scripts/live_client.py --watch SESSION_ID          # attach only
"""
from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen

from websockets.sync.client import connect  # ships with uvicorn[standard]


def main() -> int:
    parser = argparse.ArgumentParser(description="ALICE live demo client")
    parser.add_argument("--base", default="http://127.0.0.1:8710")
    parser.add_argument("--transcript")
    parser.add_argument("--video")
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--pace", type=float, default=1.0)
    parser.add_argument("--watch", help="Attach to an existing session id")
    args = parser.parse_args()

    if args.watch:
        session_id = args.watch
    else:
        source = (
            {"transcript_path": args.transcript}
            if args.transcript
            else {"video_path": args.video, "fake": args.fake}
        )
        req = Request(
            f"{args.base}/sessions",
            data=json.dumps({"source": source, "pace": args.pace}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req) as resp:
            session_id = json.loads(resp.read())["session_id"]
        print(f"session: {session_id}")

    ws_url = args.base.replace("http", "ws") + f"/sessions/{session_id}/events"
    with connect(ws_url) as ws:
        for raw in ws:
            frame = json.loads(raw)
            if "state" in frame:
                print(f"[terminal] {frame['state']}"
                      + (f" ({frame['reason']})" if frame.get("reason") else ""))
                break
            ev = frame["event"]
            recent = ev.get("recent")
            recent_s = f"{recent['composite_score']:5.1f}" if recent else "  -- "
            print(
                f"[t={ev['stream_time_seconds']:6.1f}s] seq={frame['seq']:<3} "
                f"{ev['kind']:<8} cumulative={ev['cumulative']['composite_score']:5.1f} "
                f"recent={recent_s}"
            )
    print("NOTE: anomaly signals, not ground truth. ~75% F1 ceiling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

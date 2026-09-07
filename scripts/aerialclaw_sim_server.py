#!/usr/bin/env python3
"""Run one AeroVerbs scene as a loopback AerialClaw simulator service."""

from __future__ import annotations

import argparse
import json

from aeroverbs.baselines.aerialclaw_bridge import (
    AerialClawBridgeServer,
    AeroVerbsEpisode,
    EpisodeConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    episode = AeroVerbsEpisode.from_sample(
        args.sample_dir,
        data_root=args.data_root,
        config=EpisodeConfig(max_steps=args.max_steps),
    )
    with AerialClawBridgeServer(episode, host=args.host, port=args.port) as server:
        print(json.dumps({"bridge_url": server.base_url}), flush=True)
        try:
            server._thread.join()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

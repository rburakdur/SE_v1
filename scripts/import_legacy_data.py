#!/usr/bin/env python3
"""Legacy CSV/JSON import skeleton (optional, not wired in v0.1)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Import legacy trade/signal exports into SQLite (skeleton)")
    parser.add_argument("--source", required=True, help="Path to legacy CSV/JSON")
    parser.add_argument("--kind", required=True, choices=["signals", "trades", "positions"])
    args = parser.parse_args()
    print(
        "Skeleton only. Implement mapping from legacy schema to rbdcrypt SQLite tables.",
        f"source={args.source}",
        f"kind={args.kind}",
    )


if __name__ == "__main__":
    main()

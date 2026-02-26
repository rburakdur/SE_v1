from __future__ import annotations

from enum import StrEnum


class BTCTrendFilterMode(StrEnum):
    HARD_BLOCK = "hard_block"
    SOFT_PENALTY = "soft_penalty"


class ChopPolicy(StrEnum):
    BLOCK = "block"
    PENALTY = "penalty"
    ALLOW = "allow"


class BalanceMode(StrEnum):
    CUMULATIVE = "cumulative"
    DAILY_RESET = "daily_reset"

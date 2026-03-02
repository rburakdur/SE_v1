from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


_PROFILE_EXTENSIONS: tuple[str, ...] = (".json", ".yaml", ".yml")


class HtfBiasConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeframe: str = Field(default="1h", validation_alias=AliasChoices("timeframe", "TIMEFRAME"))
    ma_type: Literal["EMA", "HMA"] = Field(
        default="EMA",
        validation_alias=AliasChoices("ma_type", "HTF_BIAS_MA_TYPE"),
    )
    period: int = Field(default=50, validation_alias=AliasChoices("period", "HTF_BIAS_PERIOD"))
    directional_filter: bool = Field(
        default=True,
        validation_alias=AliasChoices("directional_filter", "DIRECTIONAL_FILTER"),
    )

    @field_validator("period")
    @classmethod
    def _validate_period(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be > 0")
        return value


class StrategyFilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_filter: bool = Field(default=True, validation_alias=AliasChoices("session_filter", "SESSION_FILTER"))
    candidate_min: float = Field(validation_alias=AliasChoices("candidate_min", "CANDIDATE_MIN"))
    auto_min: float = Field(validation_alias=AliasChoices("auto_min", "AUTO_MIN"))
    trigger_mode: Literal["WT", "WT_HMA_COMBO", "HMA_CROSS"] = Field(
        default="WT",
        validation_alias=AliasChoices("trigger_mode", "TRIGGER_MODE"),
    )
    ltf_trigger: Literal["WT", "WT_HMA_COMBO", "HMA_CROSS"] = Field(
        default="WT",
        validation_alias=AliasChoices("ltf_trigger", "LTF_TRIGGER"),
    )
    htf_bias: HtfBiasConfig = Field(default_factory=HtfBiasConfig)
    choppiness_filter: bool = Field(
        default=False,
        validation_alias=AliasChoices("choppiness_filter", "CHOPPINESS_FILTER"),
    )
    session_no_entry_start_hour_utc: int | None = Field(
        default=None,
        validation_alias=AliasChoices("session_no_entry_start_hour_utc", "SESSION_NO_ENTRY_START_HOUR_UTC"),
    )
    session_no_entry_end_hour_utc: int | None = Field(
        default=None,
        validation_alias=AliasChoices("session_no_entry_end_hour_utc", "SESSION_NO_ENTRY_END_HOUR_UTC"),
    )

    @field_validator("candidate_min", "auto_min")
    @classmethod
    def _validate_score_threshold(cls, value: float) -> float:
        if value < 0 or value > 100:
            raise ValueError("must be between 0 and 100")
        return float(value)

    @field_validator("session_no_entry_start_hour_utc", "session_no_entry_end_hour_utc")
    @classmethod
    def _validate_hour(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value < 0 or value > 23:
            raise ValueError("must be between 0 and 23")
        return int(value)

    @model_validator(mode="after")
    def _validate_session_window(self) -> "StrategyFilterConfig":
        start = self.session_no_entry_start_hour_utc
        end = self.session_no_entry_end_hour_utc
        if (start is None) ^ (end is None):
            raise ValueError(
                "session_no_entry_start_hour_utc and session_no_entry_end_hour_utc must be provided together"
            )
        return self


class StrategyGeometryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sl_m: float = Field(validation_alias=AliasChoices("sl_m", "SL_M"))
    tp1_m: float = Field(validation_alias=AliasChoices("tp1_m", "TP1_M"))
    tp2_m: float = Field(validation_alias=AliasChoices("tp2_m", "TP2_M"))

    @field_validator("sl_m", "tp1_m", "tp2_m")
    @classmethod
    def _validate_multiple(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be > 0")
        return float(value)


class StrategyEntryPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_candidate_entries: bool = Field(
        validation_alias=AliasChoices("allow_candidate_entries", "ALLOW_CANDIDATE_ENTRIES"),
    )
    allow_auto_entries: bool = Field(
        validation_alias=AliasChoices("allow_auto_entries", "ALLOW_AUTO_ENTRIES"),
    )
    candidate_lot_multiplier: float = Field(
        validation_alias=AliasChoices("candidate_lot_multiplier", "CANDIDATE_LOT_MULTIPLIER"),
    )
    auto_lot_multiplier: float = Field(
        validation_alias=AliasChoices("auto_lot_multiplier", "AUTO_LOT_MULTIPLIER"),
    )
    symbol_cooldown_minutes: int = Field(
        default=20,
        validation_alias=AliasChoices("symbol_cooldown_minutes", "SYMBOL_COOLDOWN_MINUTES"),
    )

    @field_validator("candidate_lot_multiplier", "auto_lot_multiplier")
    @classmethod
    def _validate_multiplier(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be > 0")
        return float(value)

    @field_validator("symbol_cooldown_minutes")
    @classmethod
    def _validate_symbol_cooldown(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be >= 0")
        return int(value)


class StrategyExitPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable_session_exit: bool = Field(
        default=True,
        validation_alias=AliasChoices("enable_session_exit", "ENABLE_SESSION_EXIT"),
    )
    candidate_specific_exits: bool | dict[str, Any] = Field(
        default=False,
        validation_alias=AliasChoices("candidate_specific_exits", "CANDIDATE_SPECIFIC_EXITS"),
    )


class StrategyProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    filters: StrategyFilterConfig
    geometry: StrategyGeometryConfig
    entry_policy: StrategyEntryPolicyConfig
    exit_policy: StrategyExitPolicyConfig = Field(default_factory=StrategyExitPolicyConfig)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        val = value.strip()
        if not val:
            raise ValueError("cannot be empty")
        return val


def resolve_strategy_profile_path(
    *,
    profile_name: str,
    profiles_dir: Path,
    explicit_profile_path: Path | None = None,
) -> Path:
    if explicit_profile_path is not None:
        resolved = _resolve_candidate(explicit_profile_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Strategy profile path does not exist: {resolved}")
        return resolved

    roots: list[Path] = []
    cwd = Path.cwd().resolve()
    roots.append(cwd)

    repo_root = Path(__file__).resolve().parents[3]
    if repo_root != cwd:
        roots.append(repo_root)

    searched: list[Path] = []
    for root in roots:
        base_dir = profiles_dir if profiles_dir.is_absolute() else (root / profiles_dir)
        for ext in _PROFILE_EXTENSIONS:
            candidate = (base_dir / f"{profile_name}{ext}").resolve()
            searched.append(candidate)
            if candidate.is_file():
                return candidate

    candidate_list = ", ".join(str(path) for path in searched)
    raise FileNotFoundError(
        f"Strategy profile '{profile_name}' not found. Searched: {candidate_list}"
    )


def load_strategy_profile(
    *,
    profile_name: str,
    profiles_dir: Path,
    explicit_profile_path: Path | None = None,
) -> StrategyProfile:
    profile_path = resolve_strategy_profile_path(
        profile_name=profile_name,
        profiles_dir=profiles_dir,
        explicit_profile_path=explicit_profile_path,
    )
    payload = _load_profile_payload(profile_path)
    try:
        profile = StrategyProfile.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(_format_validation_error(profile_path, exc)) from exc

    file_stem = profile_path.stem
    if profile.name != file_stem:
        raise ValueError(
            "Strategy profile name mismatch: "
            f"name='{profile.name}' file='{file_stem}' path='{profile_path}'"
        )
    return profile


def _load_profile_payload(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON strategy profile: path='{path}' line={exc.lineno} col={exc.colno}"
            ) from exc
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ValueError(
                "YAML strategy profiles require PyYAML. Install pyyaml or use JSON profiles."
            ) from exc
        data = yaml.safe_load(raw_text)
    else:
        raise ValueError(
            f"Unsupported strategy profile extension '{suffix}'. "
            f"Use one of: {', '.join(_PROFILE_EXTENSIONS)}"
        )

    if not isinstance(data, dict):
        raise ValueError(f"Strategy profile must be a JSON/YAML object. path='{path}'")
    return data


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"Invalid strategy profile: {path}"]
    for error in exc.errors():
        location = ".".join(str(x) for x in error.get("loc", ()))
        message = error.get("msg", "validation error")
        lines.append(f"- {location}: {message}")
    return "\n".join(lines)


def _resolve_candidate(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()

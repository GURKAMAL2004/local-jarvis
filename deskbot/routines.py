"""Taught routines: `deskbot teach <name>` records a real tool-calling session
into routines/<name>.yaml, `deskbot run <name>` replays it with parameters.

A routine is a flat list of recorded tool calls with string argument values
optionally templated as "{param}". Running substitutes real values (CLI
overrides, falling back to the defaults captured at teach time) via str.format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from deskbot import paths


class RoutineNotFoundError(Exception):
    pass


@dataclass
class RoutineStep:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoutineStep":
        return cls(tool=data["tool"], args=data.get("args", {}) or {})


@dataclass
class Routine:
    name: str
    description: str
    steps: list[RoutineStep] = field(default_factory=list)
    placeholders: dict[str, str] = field(default_factory=dict)  # param name -> default value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "placeholders": self.placeholders,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "Routine":
        return cls(
            name=name,
            description=data.get("description", ""),
            steps=[RoutineStep.from_dict(s) for s in data.get("steps", [])],
            placeholders=data.get("placeholders", {}) or {},
        )

    def resolved_params(self, overrides: dict[str, str]) -> dict[str, str]:
        params = dict(self.placeholders)
        params.update(overrides)
        return params

    def resolved_steps(self, overrides: dict[str, str]) -> list[RoutineStep]:
        params = self.resolved_params(overrides)
        resolved = []
        for step in self.steps:
            args = {}
            for key, value in step.args.items():
                if isinstance(value, str):
                    try:
                        args[key] = value.format(**params)
                    except KeyError as e:
                        raise ValueError(
                            f"Routine '{self.name}' step '{step.tool}' needs parameter {e} "
                            f"— pass it with --param {e.args[0]}=<value>"
                        ) from e
                else:
                    args[key] = value
            resolved.append(RoutineStep(tool=step.tool, args=args))
        return resolved


def routine_path(name: str):
    return paths.ROUTINES_DIR / f"{name}.yaml"


def list_routines() -> list[str]:
    paths.ensure_dirs()
    return sorted(p.stem for p in paths.ROUTINES_DIR.glob("*.yaml"))


def load_routine(name: str) -> Routine:
    path = routine_path(name)
    if not path.exists():
        raise RoutineNotFoundError(
            f"No routine named '{name}'. Known routines: {', '.join(list_routines()) or '(none)'}"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Routine.from_dict(name, data)


def save_routine(routine: Routine) -> None:
    paths.ensure_dirs()
    with open(routine_path(routine.name), "w", encoding="utf-8") as f:
        yaml.safe_dump(routine.to_dict(), f, sort_keys=False, allow_unicode=True)


def delete_routine(name: str) -> None:
    path = routine_path(name)
    if not path.exists():
        raise RoutineNotFoundError(f"No routine named '{name}'.")
    path.unlink()

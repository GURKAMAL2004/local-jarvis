"""Persona loading and the interactive `deskbot persona create` wizard."""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from deskbot import paths


@dataclass
class Persona:
    name: str
    role: str = "A helpful assistant"
    tone: str = "neutral, helpful"
    greeting_style: str = "Greets the user briefly and asks how it can help"
    sample_phrases: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)
    humor_level: str = "low"
    language_mix: str = "English"

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "Persona":
        return cls(
            name=name,
            role=data.get("role", cls.role),
            tone=data.get("tone", cls.tone),
            greeting_style=data.get("greeting_style", cls.greeting_style),
            sample_phrases=data.get("sample_phrases", []) or [],
            boundaries=data.get("boundaries", []) or [],
            humor_level=data.get("humor_level", cls.humor_level),
            language_mix=data.get("language_mix", cls.language_mix),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "tone": self.tone,
            "greeting_style": self.greeting_style,
            "sample_phrases": self.sample_phrases,
            "boundaries": self.boundaries,
            "humor_level": self.humor_level,
            "language_mix": self.language_mix,
        }

    def system_prompt(self) -> str:
        lines = [
            f"You are '{self.name}', {self.role}.",
            f"Tone: {self.tone}. Humor level: {self.humor_level}.",
            f"Greeting style: {self.greeting_style}.",
            f"Preferred language mix: {self.language_mix}.",
        ]
        if self.sample_phrases:
            lines.append("Phrases you naturally use: " + "; ".join(self.sample_phrases))
        if self.boundaries:
            lines.append("Boundaries you always respect:")
            lines.extend(f"- {b}" for b in self.boundaries)
        lines.append(
            "Stay fully in character. Keep replies natural and conversational, "
            "not robotic or overly formal unless the persona calls for it."
        )
        return "\n".join(lines)


class PersonaNotFoundError(Exception):
    pass


def persona_path(name: str):
    return paths.PERSONAS_DIR / f"{name}.yaml"


def list_personas() -> list[str]:
    paths.ensure_dirs()
    return sorted(p.stem for p in paths.PERSONAS_DIR.glob("*.yaml"))


def load_persona(name: str) -> Persona:
    path = persona_path(name)
    if not path.exists():
        raise PersonaNotFoundError(
            f"No persona named '{name}'. Known personas: {', '.join(list_personas()) or '(none)'}"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Persona.from_dict(name, data)


def save_persona(persona: Persona) -> None:
    paths.ensure_dirs()
    with open(persona_path(persona.name), "w", encoding="utf-8") as f:
        yaml.safe_dump(persona.to_dict(), f, sort_keys=False, allow_unicode=True)


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or (default or "")


def _ask_list(prompt: str) -> list[str]:
    print(f"{prompt} (one per line, blank line to finish):")
    items = []
    while True:
        line = input("  - ").strip()
        if not line:
            break
        items.append(line)
    return items


def create_persona_wizard() -> Persona:
    """Interactive wizard used by `deskbot persona create`."""
    print("=== deskbot persona wizard ===")
    name = _ask("Persona name (used as -p <name>)")
    while not name:
        print("A name is required.")
        name = _ask("Persona name (used as -p <name>)")

    role = _ask("Role / who is this persona", default="A helpful assistant")
    tone = _ask("Tone (e.g. warm, formal, sarcastic)", default="warm, helpful")
    greeting_style = _ask(
        "Greeting style (how they open a conversation)",
        default="Greets the user briefly and asks how it can help",
    )
    humor_level = _ask("Humor level (none/low/medium/high)", default="low")
    language_mix = _ask(
        "Language mix (e.g. 'English', 'English + Hinglish')", default="English"
    )
    sample_phrases = _ask_list("Sample phrases this persona would say")
    boundaries = _ask_list("Boundaries this persona must always respect")

    persona = Persona(
        name=name,
        role=role,
        tone=tone,
        greeting_style=greeting_style,
        sample_phrases=sample_phrases,
        boundaries=boundaries,
        humor_level=humor_level,
        language_mix=language_mix,
    )
    save_persona(persona)
    print(f"Saved persona '{name}' -> {persona_path(name)}")
    return persona

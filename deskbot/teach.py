"""`deskbot teach <name>` — perform a task once via the normal tool-calling
agent loop, record every successful tool call, then interactively let the
user turn specific argument values into reusable {placeholders}."""

from __future__ import annotations

from rich.console import Console

from deskbot.agent import Agent
from deskbot.routines import Routine, RoutineStep, save_routine

console = Console()


def teach_routine(name: str, agent: Agent) -> Routine | None:
    console.print(f"[bold]Teaching routine '{name}'[/bold]")
    description = console.input(
        "Describe the task, as you would to a person (I'll perform it once and remember how): "
    ).strip()
    if not description:
        console.print("[red]A description is required.[/red]")
        return None

    recorded: list[RoutineStep] = []

    def record(tool_name: str, args: dict, result: dict) -> None:
        if result.get("ok", True):
            recorded.append(RoutineStep(tool=tool_name, args=dict(args)))

    agent.tools.on_call = record
    try:
        console.print()
        agent.one_shot(description)
    finally:
        agent.tools.on_call = None

    if not recorded:
        console.print(
            "\n[yellow]No tool calls were recorded, so there's nothing to save as a routine "
            "(the model answered without using any tools).[/yellow]"
        )
        return None

    console.print(f"\n[bold]Recorded {len(recorded)} step(s):[/bold]")
    for i, step in enumerate(recorded, 1):
        console.print(f"  {i}. {step.tool}({step.args})")

    placeholders: dict[str, str] = {}
    console.print(
        "\n[dim]For each argument, you can turn it into a parameter so future runs can "
        "override it (e.g. a search query or file name).[/dim]"
    )
    for i, step in enumerate(recorded, 1):
        for key, value in list(step.args.items()):
            if not isinstance(value, str):
                continue
            answer = console.input(
                f"Step {i}: {step.tool}(...) — make '{key}' (value: '{value}') a parameter? [y/N]: "
            ).strip().lower()
            if answer == "y":
                param_name = console.input(f"  Parameter name [{key}]: ").strip() or key
                placeholders[param_name] = value
                step.args[key] = f"{{{param_name}}}"

    routine = Routine(name=name, description=description, steps=recorded, placeholders=placeholders)
    save_routine(routine)
    console.print(f"\n[green]Saved routine '{name}'.[/green] Run it with: deskbot run {name}")
    if placeholders:
        example = " ".join(f"--param {k}=<value>" for k in placeholders)
        console.print(f"Override parameters with: deskbot run {name} {example}")
    return routine

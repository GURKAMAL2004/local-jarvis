"""`deskbot run <name>` — replays a taught routine step by step.

Resilience per the Phase 3 quality bar: each step already gets a timeout +
one retry inside ToolRegistry.invoke. If a step still fails, the agent
re-plans that single step once (asking the model for a corrected tool call
given the error), retries the re-planned call once, and aborts with a
readable log if that also fails — it does not silently skip or loop forever.
"""

from __future__ import annotations

import logging
from typing import Any

from rich.console import Console

from deskbot.agent import Agent
from deskbot.llm import OllamaConnectionError, OllamaModelError
from deskbot.routines import load_routine

console = Console()
logger = logging.getLogger("deskbot.routines")


def _replan_step(
    agent: Agent, description: str, tool: str, args: dict[str, Any], error: str
) -> tuple[str, dict[str, Any]] | None:
    model = agent.config.resolved_tier.text_model
    messages = [
        {
            "role": "system",
            "content": (
                "You are fixing one failed step of an automated routine. Given the original "
                "task, the tool call that failed, and the error it produced, call exactly one "
                "tool with corrected arguments. If you can't fix it, respond with plain text "
                "instead of calling a tool."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original task: {description}\n"
                f"Failed step: {tool}({args})\n"
                f"Error: {error}\n"
                "Propose a corrected tool call."
            ),
        },
    ]
    try:
        message = agent.client.chat_once(
            model, messages, temperature=agent.config.temperature, tools=agent.tools.as_ollama_tools()
        )
    except (OllamaConnectionError, OllamaModelError) as e:
        logger.warning("Re-plan LLM call failed: %s", e)
        return None
    if not message.tool_calls:
        return None
    call = message.tool_calls[0].get("function", {})
    name = call.get("name")
    if not name:
        return None
    return name, call.get("arguments") or {}


def run_routine(name: str, overrides: dict[str, str], agent: Agent) -> bool:
    routine = load_routine(name)
    steps = routine.resolved_steps(overrides)

    console.print(f"[bold]Running routine '{name}'[/bold] ({len(steps)} step(s))")
    for i, step in enumerate(steps, 1):
        console.print(f"  {i}. {step.tool}({step.args})")
        result = agent.tools.invoke(step.tool, step.args)
        if result.get("ok", True):
            continue

        console.print(
            f"     [yellow]step {i} failed: {result.get('error')} — asking the model to re-plan[/yellow]"
        )
        replanned = _replan_step(agent, routine.description, step.tool, step.args, str(result.get("error")))
        if replanned is None:
            console.print(f"[red]Routine '{name}' aborted at step {i}: could not re-plan after failure.[/red]")
            console.print(f"[red]  Original error: {result.get('error')}[/red]")
            return False

        new_tool, new_args = replanned
        console.print(f"     retrying as: {new_tool}({new_args})")
        result2 = agent.tools.invoke(new_tool, new_args)
        if not result2.get("ok", True):
            console.print(f"[red]Routine '{name}' aborted at step {i}: retry also failed.[/red]")
            console.print(f"[red]  Error: {result2.get('error')}[/red]")
            return False

    console.print(f"[green]Routine '{name}' completed successfully.[/green]")
    return True

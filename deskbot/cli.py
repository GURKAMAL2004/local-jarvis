"""deskbot's command-line entry point.

    deskbot                       interactive agent REPL (tools enabled)
    deskbot chat -p <name>        pure persona conversation, no tools
    deskbot do "<task>"           one-shot task, prints result, exits
    deskbot research "<topic>"    search, read multiple sources, and summarize a topic
    deskbot persona create        interactive persona wizard
    deskbot persona list          list known personas
    deskbot teach <name>          record a task once as a reusable routine
    deskbot run <name>            replay a taught routine
    deskbot routines list         list taught routines
    deskbot routines edit <name>  open a routine's YAML for editing
    deskbot routines delete <name> delete a routine
    deskbot schedule <name> <cron> register a routine with Windows Task Scheduler
    deskbot chess [--color white|black]  play chess against the local model
    deskbot ui [--port N]         launch the local web interface
    deskbot watch [--port N]      distraction-free YouTube kiosk (no search bar, no feed)
    deskbot doctor                environment diagnostics
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from deskbot.agent import Agent
from deskbot.config import load_config
from deskbot.logging_setup import setup_logging
from deskbot.persona import PersonaNotFoundError, create_persona_wizard, list_personas

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deskbot", description="Local Jarvis-style CLI agent")
    sub = parser.add_subparsers(dest="command")

    chat_p = sub.add_parser("chat", help="Pure persona conversation, no tools")
    chat_p.add_argument("-p", "--persona", required=True, help="Persona name")

    do_p = sub.add_parser("do", help="One-shot task, prints result, exits")
    do_p.add_argument("task", help="Task description")
    do_p.add_argument("-p", "--persona", default=None, help="Persona name (default: config default)")

    research_p = sub.add_parser(
        "research", help="Deep-research a topic: search, read multiple sources, summarize"
    )
    research_p.add_argument("topic", nargs="?", default=None, help="Topic to research (omit to be prompted)")
    research_p.add_argument(
        "--mode", choices=["quick", "standard", "deep", "relentless", "scientist", "authority"], default=None,
        help="Research depth preset. Default: interactive menu (or 'standard' with --no-menu).",
    )
    research_p.add_argument(
        "--quick-model", default=None, help="Model for follow-up/gap questions this run (overrides config)"
    )
    research_p.add_argument(
        "--synthesis-model", default=None, help="Model for writing sections/report this run (overrides config)"
    )
    research_p.add_argument(
        "--no-menu", action="store_true",
        help="Skip the interactive method/model menu; use --mode (default 'standard') and config defaults",
    )

    persona_p = sub.add_parser("persona", help="Manage personas")
    persona_sub = persona_p.add_subparsers(dest="persona_command", required=True)
    persona_sub.add_parser("create", help="Interactive persona creation wizard")
    persona_sub.add_parser("list", help="List known personas")

    teach_p = sub.add_parser("teach", help="Record a task once as a reusable routine")
    teach_p.add_argument("name", help="Routine name")

    run_p = sub.add_parser("run", help="Replay a taught routine")
    run_p.add_argument("name", help="Routine name")
    run_p.add_argument(
        "--param", action="append", default=[], metavar="k=v",
        help="Override a routine parameter, e.g. --param query='new search'",
    )

    routines_p = sub.add_parser("routines", help="Manage taught routines")
    routines_sub = routines_p.add_subparsers(dest="routines_command", required=True)
    routines_sub.add_parser("list", help="List taught routines")
    edit_p = routines_sub.add_parser("edit", help="Open a routine's YAML for editing")
    edit_p.add_argument("name")
    delete_p = routines_sub.add_parser("delete", help="Delete a routine")
    delete_p.add_argument("name")

    schedule_p = sub.add_parser("schedule", help="Register a routine with Windows Task Scheduler")
    schedule_p.add_argument("name", help="Routine name")
    schedule_p.add_argument("cron", help="5-field cron subset, e.g. '0 9 * * *' for daily at 9am")

    chess_p = sub.add_parser("chess", help="Play chess against the local model in the terminal")
    chess_p.add_argument("--color", choices=["white", "black"], default="white", help="Which side you play")

    ui_p = sub.add_parser("ui", help="Launch the local web interface")
    ui_p.add_argument("--port", type=int, default=8420, help="Port to serve on (default: 8420)")
    ui_p.add_argument("--no-browser", action="store_true", help="Don't automatically open a browser tab")

    watch_p = sub.add_parser(
        "watch", help="Distraction-free YouTube kiosk: tell it what to watch, it asks, then plays — no search bar, no feed"
    )
    watch_p.add_argument("--port", type=int, default=8421, help="Local port for the kiosk backend (default: 8421)")

    sub.add_parser("doctor", help="Diagnose the local environment")

    return parser


def _parse_param_overrides(pairs: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--param must be in the form k=v, got: {pair}")
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value
    return overrides


def _force_utf8_console() -> None:
    """Windows terminals often default stdout/stderr to the legacy cp1252
    codepage, which can't encode characters like '→' (used in tool-call
    logging). Reconfiguring in place fixes every Console() instance across
    the app, since they all bind to this same sys.stdout/stderr object."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_config()
    setup_logging(config.log_level)

    if args.command is None:
        from deskbot.tools import build_tool_registry

        tools = build_tool_registry(config)
        agent = Agent(config, tools=tools)
        try:
            agent.chat_repl(config.default_persona, tools_enabled=True)
        finally:
            tools.browser_session.close()
        return 0

    if args.command == "chat":
        try:
            agent = Agent(config)
            agent.chat_repl(args.persona, tools_enabled=False)
        except PersonaNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return 1
        return 0

    if args.command == "do":
        from deskbot.tools import build_tool_registry

        tools = build_tool_registry(config)
        try:
            agent = Agent(config, tools=tools)
            agent.one_shot(args.task, persona_name=args.persona)
        except PersonaNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return 1
        finally:
            tools.browser_session.close()
        return 0

    if args.command == "research":
        import os
        from dataclasses import replace

        from deskbot.research import RESEARCH_MODE_PRESETS, prompt_research_setup, run_deep_research
        from deskbot.tools import build_tool_registry

        tools = build_tool_registry(config)
        try:
            agent = Agent(config, tools=tools)

            topic = args.topic
            if topic is None:
                topic = input("What do you want to research? ").strip()
                if not topic:
                    console.print("[red]No topic given.[/red]")
                    return 1

            explicit_flags = args.mode or args.quick_model or args.synthesis_model or args.no_menu
            # Never show the interactive menu on a non-interactive stdin (e.g.
            # a scheduled/`deskbot run` invocation) — it would just hang.
            if not explicit_flags and sys.stdin.isatty():
                options = prompt_research_setup(agent)
            else:
                options = replace(RESEARCH_MODE_PRESETS[args.mode or "standard"])
                if args.quick_model:
                    options.quick_model = args.quick_model
                if args.synthesis_model:
                    options.synthesis_model = args.synthesis_model

            result = run_deep_research(agent, topic, options=options)
        finally:
            tools.browser_session.close()

        if result.saved_path is None:
            return 1
        try:
            os.startfile(str(result.saved_path))  # noqa: S606 - open the report for the user
        except OSError:
            pass
        return 0

    if args.command == "persona":
        if args.persona_command == "create":
            create_persona_wizard()
            return 0
        if args.persona_command == "list":
            names = list_personas()
            if not names:
                console.print("[dim]No personas yet. Run: deskbot persona create[/dim]")
            for n in names:
                console.print(f"- {n}")
            return 0

    if args.command == "teach":
        from deskbot.teach import teach_routine
        from deskbot.tools import build_tool_registry

        tools = build_tool_registry(config)
        try:
            agent = Agent(config, tools=tools)
            teach_routine(args.name, agent)
        finally:
            tools.browser_session.close()
        return 0

    if args.command == "run":
        from deskbot.routine_runner import run_routine
        from deskbot.routines import RoutineNotFoundError
        from deskbot.tools import build_tool_registry

        try:
            overrides = _parse_param_overrides(args.param)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            return 1

        tools = build_tool_registry(config)
        try:
            agent = Agent(config, tools=tools)
            ok = run_routine(args.name, overrides, agent)
        except RoutineNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return 1
        except ValueError as e:  # missing required param
            console.print(f"[red]{e}[/red]")
            return 1
        finally:
            tools.browser_session.close()
        return 0 if ok else 1

    if args.command == "routines":
        from deskbot.routines import (
            RoutineNotFoundError,
            delete_routine,
            list_routines,
            routine_path,
        )

        if args.routines_command == "list":
            names = list_routines()
            if not names:
                console.print("[dim]No routines yet. Run: deskbot teach <name>[/dim]")
            for n in names:
                console.print(f"- {n}")
            return 0

        if args.routines_command == "edit":
            import os

            path = routine_path(args.name)
            if not path.exists():
                console.print(f"[red]No routine named '{args.name}'.[/red]")
                return 1
            console.print(f"Opening {path}")
            os.startfile(str(path))  # noqa: S606 - intentional, opens in the user's default editor
            return 0

        if args.routines_command == "delete":
            try:
                delete_routine(args.name)
            except RoutineNotFoundError as e:
                console.print(f"[red]{e}[/red]")
                return 1
            console.print(f"[green]Deleted routine '{args.name}'.[/green]")
            return 0

    if args.command == "schedule":
        from deskbot.scheduler import UnsupportedScheduleError, schedule_routine
        from deskbot.routines import RoutineNotFoundError, load_routine

        try:
            load_routine(args.name)  # fail fast if the routine doesn't exist
        except RoutineNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return 1

        try:
            result = schedule_routine(args.name, args.cron)
        except UnsupportedScheduleError as e:
            console.print(f"[red]{e}[/red]")
            return 1

        if result["ok"]:
            console.print(f"[green]Scheduled '{args.name}' as Windows task '{result['task_name']}'.[/green]")
        else:
            console.print(f"[red]schtasks failed:[/red]\n{result['stderr'] or result['stdout']}")
        return 0 if result["ok"] else 1

    if args.command == "chess":
        from deskbot.chess_game import play_chess

        agent = Agent(config)
        play_chess(agent, human_color=args.color)
        return 0

    if args.command == "ui":
        import threading
        import webbrowser

        import uvicorn

        from deskbot.webui.server import create_app

        url = f"http://127.0.0.1:{args.port}"
        console.print(f"[bold]deskbot web UI:[/bold] {url}")
        if not args.no_browser:
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        uvicorn.run(create_app(config), host="127.0.0.1", port=args.port, log_level="warning")
        return 0

    if args.command == "watch":
        from deskbot.watch_kiosk import run_watch_kiosk

        return run_watch_kiosk(config, port=args.port)

    if args.command == "doctor":
        from deskbot.doctor import run_doctor

        ok = run_doctor()
        return 0 if ok else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

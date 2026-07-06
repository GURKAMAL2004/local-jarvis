"""Builds the populated ToolRegistry used by the tool-enabled REPL and
`deskbot do`. `deskbot chat -p <persona>` deliberately uses an empty
ToolRegistry instead (pure persona conversation, no tools)."""

from __future__ import annotations

from deskbot.config import Config


def build_tool_registry(config: Config):
    from deskbot.agent import ToolRegistry
    from deskbot.tools.apps import OPEN_APP_SCHEMA, make_open_app
    from deskbot.tools.browser import BrowserSession, register_browser_tools
    from deskbot.tools.shell import RUN_SHELL_SCHEMA, make_run_shell

    registry = ToolRegistry()
    registry.register("run_shell", make_run_shell(config), RUN_SHELL_SCHEMA)
    registry.register("open_app", make_open_app(), OPEN_APP_SCHEMA)

    session = BrowserSession(config)
    register_browser_tools(registry, session)
    registry.browser_session = session  # type: ignore[attr-defined]  # closed by cli.py on exit
    return registry

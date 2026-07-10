"""FastAPI backend for `deskbot ui` — a local web interface wrapping the
existing CLI features (chat, deep research, chess, routines) plus a premium
unlock panel. Runs on 127.0.0.1 only; never talks to anything but Ollama and
your own browser.

Chat and chess run in-process (fast, turn-based — reuses Agent/chess_game
directly). Research and routine runs go through webui.jobs as real
subprocesses of `python -m deskbot ...`, since those are long-running and
need a genuine stoppable process — see jobs.py's docstring.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import chess
import qrcode
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from deskbot import paths
from deskbot.agent import Agent
from deskbot.chess_game import _game_status, _legal_moves_san, choose_model_move, render_board
from deskbot.config import Config, load_config
from deskbot.llm import OllamaConnectionError, OllamaModelError
from deskbot.memory import Memory
from deskbot.persona import PersonaNotFoundError, list_personas, load_persona
from deskbot.routines import RoutineNotFoundError, list_routines, load_routine
from deskbot.webui import jobs
from deskbot.webui import watch as watch_module
from deskbot.webui.licensing import verify_license_code

STATIC_DIR = Path(__file__).resolve().parent / "static"
PREMIUM_PATH = paths.HOME_DIR / "premium.json"

# Public receiving address for premium support/subscriptions — shown as text
# and as a QR code in the Premium panel. It's just a public wallet address;
# safe to ship. See licensing.py for how a payment turns into an unlock code.
PREMIUM_WALLET_ADDRESS = "0x2Fe6cB50E5a3515a16266bcBB8A8a5f5936E1829"
PREMIUM_CONTACT_EMAIL = "gkmander7@gmail.com"

app = FastAPI(title="deskbot")

_chat_sessions: dict[str, Agent] = {}
_chess_sessions: dict[str, dict[str, Any]] = {}
_chess_ai_agent: Agent | None = None


def _get_chess_agent() -> Agent:
    global _chess_ai_agent
    if _chess_ai_agent is None:
        _chess_ai_agent = Agent(_cfg())
    return _chess_ai_agent


def create_app(config: Config | None = None) -> FastAPI:
    app.state.config = config or load_config()
    return app


def _cfg() -> Config:
    return app.state.config


# --- static frontend -------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# --- system status (control panel) ------------------------------------------


@app.get("/api/status")
def api_status() -> dict:
    """Powers the Home control panel's status strip: is Ollama actually
    reachable right now, and what's pulled — the same live check `deskbot
    doctor` does, surfaced where a non-technical user will actually see it
    instead of a terminal command they don't know to run."""
    from deskbot.llm import OllamaClient

    client = OllamaClient(host=_cfg().ollama_host)
    try:
        models = client.list_models()
        return {"ollama": "up", "models": models}
    except OllamaConnectionError:
        return {"ollama": "down", "models": []}


# --- personas / chat ---------------------------------------------------------


@app.get("/api/personas")
def api_list_personas() -> dict:
    return {"personas": list_personas(), "default": _cfg().default_persona}


class ChatRequest(BaseModel):
    persona: str
    message: str


@app.post("/api/chat")
def api_chat(payload: ChatRequest) -> dict:
    try:
        load_persona(payload.persona)
    except PersonaNotFoundError as e:
        raise HTTPException(404, str(e)) from e

    agent = _chat_sessions.get(payload.persona)
    if agent is None:
        agent = Agent(_cfg(), memory=Memory())
        _chat_sessions[payload.persona] = agent

    try:
        reply = agent.converse(payload.persona, payload.message)
    except (OllamaConnectionError, OllamaModelError) as e:
        raise HTTPException(502, f"Local model error: {e}") from e
    return {"reply": reply}


# --- watch kiosk (distraction-free YouTube, no search bar/feed) ------------


@app.get("/watch", response_class=HTMLResponse)
def watch_page() -> str:
    return (STATIC_DIR / "watch.html").read_text(encoding="utf-8")


class WatchStartRequest(BaseModel):
    request: str


@app.post("/api/watch/start")
def api_watch_start(payload: WatchStartRequest) -> dict:
    session_id = watch_module.create_session(_cfg())
    session = watch_module.get_session(session_id)
    assert session is not None  # just created above
    result = session.start(payload.request)
    return {"session_id": session_id, **result}


class WatchMessageRequest(BaseModel):
    session_id: str
    message: str


@app.post("/api/watch/message")
def api_watch_message(payload: WatchMessageRequest) -> dict:
    session = watch_module.get_session(payload.session_id)
    if session is None:
        raise HTTPException(404, "Unknown watch session — start a new one")
    return session.reply(payload.message)


class WatchCommandRequest(BaseModel):
    text: str
    panes: list[dict]
    last_active_pane: int | None = None
    layout: int


@app.post("/api/watch/command")
def api_watch_command(payload: WatchCommandRequest) -> dict:
    actions = watch_module.classify_command(
        payload.text, payload.panes, payload.last_active_pane, payload.layout, _cfg()
    )
    return {"actions": actions}


# --- deep research (subprocess job) -----------------------------------------


class ResearchStartRequest(BaseModel):
    topic: str
    mode: str = "standard"
    quick_model: str | None = None
    synthesis_model: str | None = None


@app.post("/api/research/start")
def api_research_start(payload: ResearchStartRequest) -> dict:
    args = ["research", payload.topic, "--no-menu", "--mode", payload.mode]
    if payload.quick_model:
        args += ["--quick-model", payload.quick_model]
    if payload.synthesis_model:
        args += ["--synthesis-model", payload.synthesis_model]
    label = f"Research: {payload.topic} ({payload.mode})"
    job_id = jobs.start_job(jobs.deskbot_command(*args), label=label)
    return {"job_id": job_id}


class RoutineRunRequest(BaseModel):
    name: str
    params: dict[str, str] = {}


@app.post("/api/routines/run")
def api_routines_run(payload: RoutineRunRequest) -> dict:
    args = ["run", payload.name]
    for key, value in payload.params.items():
        args += ["--param", f"{key}={value}"]
    job_id = jobs.start_job(jobs.deskbot_command(*args), label=f"Routine: {payload.name}")
    return {"job_id": job_id}


@app.get("/api/jobs")
def api_jobs_list() -> dict:
    """Backs the control panel's Running Jobs list — every job started this
    server session, running or finished, so nothing can keep burning CPU
    unnoticed the way the background research job that overheated this
    machine once did."""
    return {"jobs": jobs.list_jobs()}


@app.get("/api/jobs/{job_id}/stream")
def api_job_stream(job_id: str) -> StreamingResponse:
    if jobs.get_job(job_id) is None:
        raise HTTPException(404, "Unknown job")

    def event_source():
        job = jobs.get_job(job_id)
        line_queue = job["queue"]
        while True:
            line = line_queue.get()
            if line is None:
                yield "event: done\ndata: {}\n\n"
                break
            yield f"data: {json.dumps(line)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/api/jobs/{job_id}/stop")
def api_job_stop(job_id: str) -> dict:
    if not jobs.stop_job(job_id):
        raise HTTPException(404, "Unknown job")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/report")
def api_job_report(job_id: str) -> dict:
    report = jobs.get_report(job_id)
    if report is None:
        raise HTTPException(404, "No report available yet")
    return report


# --- routines list -----------------------------------------------------------


@app.get("/api/routines")
def api_routines_list() -> dict:
    names = list_routines()
    detail = []
    for name in names:
        try:
            routine = load_routine(name)
            detail.append({"name": name, "params": sorted(routine.placeholders.keys())})
        except RoutineNotFoundError:
            continue
    return {"routines": detail}


# --- chess (in-process) -------------------------------------------------------


class ChessNewRequest(BaseModel):
    color: str = "white"


@app.post("/api/chess/new")
def api_chess_new(payload: ChessNewRequest) -> dict:
    import uuid

    session_id = uuid.uuid4().hex
    board = chess.Board()
    human_is_white = payload.color.lower().startswith("w")
    _chess_sessions[session_id] = {"board": board, "moves": [], "human_is_white": human_is_white}
    return _chess_state(session_id)


def _chess_state(session_id: str) -> dict:
    session = _chess_sessions[session_id]
    board: chess.Board = session["board"]
    return {
        "session_id": session_id,
        "board": render_board(board, perspective_white=session["human_is_white"]),
        "fen": board.fen(),
        "turn": "white" if board.turn == chess.WHITE else "black",
        "human_is_white": session["human_is_white"],
        "legal_moves": _legal_moves_san(board),
        "moves_played": session["moves"],
        "status": _game_status(board),
        "game_over": board.is_game_over(claim_draw=True),
        "result": board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else None,
    }


class ChessMoveRequest(BaseModel):
    session_id: str
    move: str


@app.post("/api/chess/move")
def api_chess_move(payload: ChessMoveRequest) -> dict:
    session = _chess_sessions.get(payload.session_id)
    if session is None:
        raise HTTPException(404, "Unknown chess session — start a new game")
    board: chess.Board = session["board"]

    # Accept both a UCI square-pair (what the clickable board sends, e.g.
    # "e2e4") and SAN (for a future text-entry fallback).
    move = None
    try:
        candidate = chess.Move.from_uci(payload.move)
        if candidate in board.legal_moves:
            move = candidate
    except ValueError:
        pass
    if move is None:
        try:
            move = board.parse_san(payload.move)
        except ValueError as e:
            raise HTTPException(400, f"'{payload.move}' isn't a legal move") from e
    san = board.san(move)
    board.push(move)
    session["moves"].append(san)

    if not board.is_game_over(claim_draw=True):
        model_san = choose_model_move(_get_chess_agent(), board, session["moves"])
        board.push(board.parse_san(model_san))
        session["moves"].append(model_san)

    return _chess_state(payload.session_id)


# --- premium -------------------------------------------------------------------


@app.get("/api/premium/wallet")
def api_premium_wallet() -> dict:
    return {"address": PREMIUM_WALLET_ADDRESS, "email": PREMIUM_CONTACT_EMAIL}


@app.get("/api/premium/qrcode.png")
def api_premium_qrcode() -> Response:
    img = qrcode.make(PREMIUM_WALLET_ADDRESS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


class PremiumVerifyRequest(BaseModel):
    email: str
    code: str


@app.post("/api/premium/verify")
def api_premium_verify(payload: PremiumVerifyRequest) -> dict:
    if not verify_license_code(payload.email, payload.code):
        raise HTTPException(400, "That code doesn't match that email — double-check both.")
    paths.ensure_dirs()
    PREMIUM_PATH.write_text(json.dumps({"email": payload.email, "code": payload.code}), encoding="utf-8")
    return {"ok": True}


@app.get("/api/premium/status")
def api_premium_status() -> dict:
    if not PREMIUM_PATH.exists():
        return {"unlocked": False}
    try:
        data = json.loads(PREMIUM_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"unlocked": False}
    unlocked = verify_license_code(data.get("email", ""), data.get("code", ""))
    return {"unlocked": unlocked, "email": data.get("email") if unlocked else None}

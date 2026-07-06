from __future__ import annotations

import chess

from deskbot.agent import Agent, ToolRegistry
from deskbot.chess_game import (
    _game_status,
    _legal_moves_san,
    _resolve_chess_model,
    choose_model_move,
    render_board,
)
from deskbot.config import load_config
from deskbot.llm import OllamaClient, OllamaConnectionError
from deskbot.memory import Memory


def _agent() -> Agent:
    return Agent(load_config(), memory=Memory(), tools=ToolRegistry())


def test_render_board_shows_starting_position_white_perspective():
    board = chess.Board()
    rendered = render_board(board)
    lines = rendered.splitlines()
    assert lines[0].startswith("8 ")
    assert lines[-1] == "   a b c d e f g h"
    assert "r n b q k b n r" in lines[0]
    assert "P P P P P P P P" in lines[-3]  # rank 2, one line above the back rank


def test_render_board_flips_for_black_perspective():
    board = chess.Board()
    rendered = render_board(board, perspective_white=False)
    lines = rendered.splitlines()
    assert lines[0].startswith("1 ")
    assert lines[-1] == "   h g f e d c b a"


def test_render_board_uses_plain_ascii_not_unicode():
    """Regression guard: plain cmd.exe's default cp1252 codepage can't
    encode chess Unicode glyphs and raises UnicodeEncodeError on print()."""
    board = chess.Board()
    rendered = render_board(board)
    rendered.encode("cp1252")  # must not raise


def test_legal_moves_san_matches_starting_position_count():
    board = chess.Board()
    assert len(_legal_moves_san(board)) == 20


def test_game_status_reports_checkmate():
    board = chess.Board()
    for move in ["f3", "e5", "g4", "Qh4#"]:  # fool's mate
        board.push_san(move)
    assert _game_status(board) == "Checkmate — Black wins."


def test_game_status_reports_check_without_mate():
    board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")  # rook checks along the e-file, king can step aside
    assert board.is_check()
    assert not board.is_checkmate()
    assert _game_status(board) == "Check."


def test_game_status_empty_for_ordinary_position():
    board = chess.Board()
    assert _game_status(board) == ""


def test_resolve_chess_model_falls_back_to_ram_tier():
    agent = _agent()
    assert _resolve_chess_model(agent) == agent.config.resolved_tier.text_model


def test_resolve_chess_model_uses_configured_model():
    agent = _agent()
    agent.config._raw.setdefault("chess", {})["model"] = "my-chess-model"
    assert _resolve_chess_model(agent) == "my-chess-model"


def test_choose_model_move_returns_forced_move_without_calling_model(monkeypatch):
    """A position with exactly one legal move shouldn't waste a model call —
    also proves the game can never get stuck: there's always a right answer."""
    called = []
    monkeypatch.setattr(
        OllamaClient, "chat",
        lambda self, model, messages, temperature=0.4, tools=None: called.append(1) or "irrelevant",
    )
    board = chess.Board()
    # A constructed position with exactly one legal move (king must escape check).
    board.set_fen("7k/8/8/8/8/8/6q1/7K w - - 0 1")
    legal = _legal_moves_san(board)
    assert len(legal) == 1

    agent = _agent()
    move = choose_model_move(agent, board, moves_played=[], model="whatever")
    assert move == legal[0]
    assert called == []


def test_choose_model_move_accepts_a_valid_model_reply(monkeypatch):
    board = chess.Board()
    legal = _legal_moves_san(board)

    monkeypatch.setattr(
        OllamaClient, "chat",
        lambda self, model, messages, temperature=0.4, tools=None: legal[3],
    )
    agent = _agent()
    move = choose_model_move(agent, board, moves_played=[], model="whatever")
    assert move == legal[3]


def test_choose_model_move_falls_back_to_random_legal_move_after_bad_replies(monkeypatch):
    board = chess.Board()
    legal = _legal_moves_san(board)

    monkeypatch.setattr(
        OllamaClient, "chat",
        lambda self, model, messages, temperature=0.4, tools=None: "not a real move at all",
    )
    agent = _agent()
    move = choose_model_move(agent, board, moves_played=[], model="whatever")
    assert move in legal  # never stalls the game even on a consistently-wrong model


def test_choose_model_move_falls_back_when_model_unreachable(monkeypatch):
    board = chess.Board()
    legal = _legal_moves_san(board)

    def raise_error(self, model, messages, temperature=0.4, tools=None):
        raise OllamaConnectionError("down")

    monkeypatch.setattr(OllamaClient, "chat", raise_error)
    agent = _agent()
    move = choose_model_move(agent, board, moves_played=[], model="whatever")
    assert move in legal

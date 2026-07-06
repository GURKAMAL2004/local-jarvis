"""`deskbot chess` — play chess against the local model in the terminal.

Same design philosophy as research.py: don't trust a small local model with
something it's unreliable at. Asked to freely generate a chess move in SAN
notation, small models regularly hallucinate illegal ones — wrong piece,
moving through a blocker, ignoring check, a square that doesn't exist. So
move *legality* is entirely code-driven via the `chess` library (python-chess),
and the model is never asked to invent a move: it's shown the exact list of
currently-legal moves for the position and told to pick one, the same way
research.py grounds the model in real search-result links instead of invented
ones. If it still returns something not on the list (retried once with a
nudge), a random legal move is played instead so a flaky model never stalls
the game.
"""

from __future__ import annotations

import logging
import random

import chess

from deskbot.agent import Agent
from deskbot.llm import OllamaConnectionError, OllamaModelError

logger = logging.getLogger("deskbot.chess")

_QUIT_WORDS = {"resign", "quit", "exit"}


def render_board(board: chess.Board, perspective_white: bool = True) -> str:
    """Plain-ASCII board (uppercase = White, lowercase = Black, '.' = empty)
    for a command-prompt window. Deliberately plain ASCII, not Unicode chess
    glyphs — plain cmd.exe's default cp1252 codepage can't encode those and
    raises UnicodeEncodeError on print(). perspective_white=False flips the
    board so the human's own pieces are always shown at the bottom,
    regardless of which color they're playing."""
    ranks = range(8, 0, -1) if perspective_white else range(1, 9)
    files = range(8) if perspective_white else range(7, -1, -1)
    lines = []
    for rank in ranks:
        row = [f"{rank} "]
        for file in files:
            piece = board.piece_at(chess.square(file, rank - 1))
            row.append(piece.symbol() if piece else ".")
        lines.append(" ".join(row))
    file_letters = "abcdefgh" if perspective_white else "hgfedcba"
    lines.append("   " + " ".join(file_letters))
    return "\n".join(lines)


def _legal_moves_san(board: chess.Board) -> list[str]:
    return [board.san(m) for m in board.legal_moves]


def _game_status(board: chess.Board) -> str:
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"Checkmate — {winner} wins."
    if board.is_stalemate():
        return "Stalemate — draw."
    if board.is_insufficient_material():
        return "Draw — insufficient material."
    if board.can_claim_fifty_moves():
        return "Draw — fifty-move rule."
    if board.can_claim_threefold_repetition():
        return "Draw — threefold repetition."
    if board.is_check():
        return "Check."
    return ""


def _resolve_chess_model(agent: Agent) -> str:
    return agent.config.get("chess", "model", default=None) or agent.config.resolved_tier.text_model


def choose_model_move(agent: Agent, board: chess.Board, moves_played: list[str], model: str | None = None) -> str:
    """Grounded move selection: the model picks from the real legal-move
    list rather than generating one, so the game can never get stuck on an
    illegal move regardless of model size."""
    model = model or _resolve_chess_model(agent)
    legal = _legal_moves_san(board)
    if len(legal) == 1:
        return legal[0]  # forced move — no need to even ask

    turn = "White" if board.turn == chess.WHITE else "Black"
    history = " ".join(moves_played) or "(none yet)"
    messages = [
        {
            "role": "system",
            "content": (
                "You are playing a chess game. You will be shown the current position "
                "and a list of every legal move. Reply with EXACTLY one move copied "
                "verbatim from that list — no explanation, no extra words, no "
                "punctuation. All out-of-check, non-hanging moves are legal to pick; "
                "prefer ones that develop pieces, control the center, protect your "
                "king, or win material/checkmate when available."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Position (FEN): {board.fen()}\n\n{render_board(board)}\n\n"
                f"Moves so far: {history}\n\n{turn} to move. "
                f"Legal moves: {', '.join(legal)}\n\nYour move:"
            ),
        },
    ]
    for _attempt in range(2):
        try:
            reply = agent.client.chat(model, messages, temperature=0.7)
        except (OllamaConnectionError, OllamaModelError) as e:
            logger.warning("chess move request failed: %s", e)
            break
        candidate = reply.strip().splitlines()[0].strip().rstrip(".") if reply.strip() else ""
        if candidate in legal:
            return candidate
        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {
                "role": "user",
                "content": f"'{candidate}' is not one of the legal moves. "
                f"Reply with exactly one move from: {', '.join(legal)}",
            }
        )

    fallback = random.choice(legal)
    logger.info("model failed to choose a legal move — playing random legal move '%s'", fallback)
    return fallback


def play_chess(agent: Agent, human_color: str = "white") -> None:
    """Interactive terminal game loop. Blocks until checkmate/stalemate/draw
    or the human resigns."""
    board = chess.Board()
    moves_played: list[str] = []
    human_is_white = human_color.lower().startswith("w")
    model = _resolve_chess_model(agent)

    print("=== deskbot chess ===")
    print(f"You are playing {'White' if human_is_white else 'Black'} against '{model}'.")
    print("Type a move in standard algebraic notation (e.g. e4, Nf3, O-O).")
    print("Other commands: 'moves' to list legal moves, 'resign'/'quit' to leave.\n")

    while not board.is_game_over(claim_draw=True):
        print(render_board(board, perspective_white=human_is_white))
        status = _game_status(board)
        if status:
            print(status)

        human_turn = (board.turn == chess.WHITE) == human_is_white
        if human_turn:
            raw = input("\nYour move: ").strip()
            if raw.lower() in _QUIT_WORDS:
                print("You resigned. Good game!")
                return
            if raw.lower() == "moves":
                print("Legal moves: " + ", ".join(_legal_moves_san(board)) + "\n")
                continue
            try:
                move = board.parse_san(raw)
            except ValueError:
                print(f"'{raw}' isn't a legal move. Type 'moves' to see your options.\n")
                continue
            san = board.san(move)
            board.push(move)
            moves_played.append(san)
        else:
            print("\ndeskbot is thinking...")
            san = choose_model_move(agent, board, moves_played, model)
            board.push(board.parse_san(san))
            moves_played.append(san)
            print(f"deskbot plays: {san}")
        print()

    print(render_board(board, perspective_white=human_is_white))
    result = board.result(claim_draw=True)
    summary = _game_status(board) or "Draw."
    print(f"\nGame over ({result}): {summary}")

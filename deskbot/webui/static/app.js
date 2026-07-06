// deskbot web UI — plain vanilla JS, no build step, no CDN dependencies.
// Talks only to the local FastAPI server on the same origin.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function showToast(message, isError = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("show");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => toast.classList.remove("show"), 4000);
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

// --- navigation --------------------------------------------------------

function goto(view) {
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  $$(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
}

$$(".nav-btn").forEach((btn) => btn.addEventListener("click", () => goto(btn.dataset.view)));
$$("[data-goto]").forEach((el) => el.addEventListener("click", () => goto(el.dataset.goto)));

// --- chat ----------------------------------------------------------------

async function loadPersonas() {
  const data = await api("/api/personas");
  const select = $("#chat-persona");
  select.innerHTML = "";
  for (const name of data.personas) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name === data.default) opt.selected = true;
    select.appendChild(opt);
  }
  if (data.personas.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "(no personas yet — run: deskbot persona create)";
    select.appendChild(opt);
  }
}

function appendChatBubble(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  bubble.textContent = text;
  $("#chat-log").appendChild(bubble);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
}

async function sendChat() {
  const input = $("#chat-input");
  const message = input.value.trim();
  if (!message) return;
  const persona = $("#chat-persona").value;
  input.value = "";
  appendChatBubble("user", message);
  const sendBtn = $("#chat-send");
  sendBtn.disabled = true;
  try {
    const data = await api("/api/chat", { method: "POST", body: JSON.stringify({ persona, message }) });
    appendChatBubble("assistant", data.reply);
  } catch (e) {
    showToast(`Chat failed: ${e.message}`, true);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

$("#chat-send").addEventListener("click", sendChat);
$("#chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendChat();
});

// --- deep research ---------------------------------------------------------

let currentResearchJob = null;

function researchLog(line) {
  const box = $("#research-console");
  box.textContent += line + "\n";
  box.scrollTop = box.scrollHeight;
}

async function startResearch() {
  const topic = $("#research-topic").value.trim();
  if (!topic) {
    showToast("Type a topic first.", true);
    return;
  }
  const mode = $("#research-mode").value;
  const quick_model = $("#research-quick-model").value.trim() || null;
  const synthesis_model = $("#research-synthesis-model").value.trim() || null;

  $("#research-progress-card").style.display = "block";
  $("#research-report-card").style.display = "none";
  $("#research-console").textContent = "";
  $("#research-status").textContent = "Starting...";
  $("#research-start").disabled = true;
  $("#research-stop").style.display = "inline-flex";

  try {
    const data = await api("/api/research/start", {
      method: "POST",
      body: JSON.stringify({ topic, mode, quick_model, synthesis_model }),
    });
    currentResearchJob = data.job_id;
    streamJob(data.job_id, {
      onLine: researchLog,
      onDone: async () => {
        $("#research-status").textContent = "Done.";
        $("#research-start").disabled = false;
        $("#research-stop").style.display = "none";
        try {
          const report = await api(`/api/jobs/${data.job_id}/report`);
          $("#research-report").textContent = report.content;
          $("#research-report-card").style.display = "block";
        } catch (_) {
          $("#research-status").textContent = "Finished, but no report file was found (check the log above).";
        }
      },
    });
  } catch (e) {
    showToast(`Could not start research: ${e.message}`, true);
    $("#research-start").disabled = false;
    $("#research-stop").style.display = "none";
  }
}

async function stopResearch() {
  if (!currentResearchJob) return;
  $("#research-status").textContent = "Stopping — writing up what's been found so far...";
  try {
    await api(`/api/jobs/${currentResearchJob}/stop`, { method: "POST" });
  } catch (e) {
    showToast(`Could not stop: ${e.message}`, true);
  }
}

function streamJob(jobId, { onLine, onDone }) {
  const source = new EventSource(`/api/jobs/${jobId}/stream`);
  source.onmessage = (event) => onLine(JSON.parse(event.data));
  source.addEventListener("done", () => {
    source.close();
    onDone();
  });
  source.onerror = () => {
    source.close();
    onDone();
  };
}

$("#research-start").addEventListener("click", startResearch);
$("#research-stop").addEventListener("click", stopResearch);

// --- routines ----------------------------------------------------------

async function loadRoutines() {
  const container = $("#routines-list");
  try {
    const data = await api("/api/routines");
    if (data.routines.length === 0) {
      container.innerHTML = '<p class="status-line">No routines taught yet. Run <code>deskbot teach &lt;name&gt;</code> in a terminal first.</p>';
      return;
    }
    container.innerHTML = "";
    for (const routine of data.routines) {
      const row = document.createElement("div");
      row.style.marginBottom = "1.25rem";
      const paramInputs = routine.params.map(
        (p) => `<input type="text" data-param="${p}" placeholder="${p}" style="max-width:220px; display:inline-block; margin-right:0.5rem;">`
      ).join("");
      row.innerHTML = `
        <h4 style="margin-bottom:0.3rem">${routine.name}</h4>
        ${routine.params.length ? `<div style="margin-bottom:0.5rem">${paramInputs}</div>` : ""}
        <button data-run="${routine.name}">Run</button>
      `;
      container.appendChild(row);
    }
    container.querySelectorAll("[data-run]").forEach((btn) => {
      btn.addEventListener("click", () => runRoutine(btn, btn.dataset.run));
    });
  } catch (e) {
    container.innerHTML = `<p class="status-line">Could not load routines: ${e.message}</p>`;
  }
}

function routineLog(line) {
  const box = $("#routines-console");
  box.textContent += line + "\n";
  box.scrollTop = box.scrollHeight;
}

async function runRoutine(button, name) {
  const row = button.closest("div");
  const params = {};
  row.querySelectorAll("[data-param]").forEach((input) => {
    if (input.value.trim()) params[input.dataset.param] = input.value.trim();
  });
  $("#routines-progress-card").style.display = "block";
  $("#routines-console").textContent = "";
  button.disabled = true;
  try {
    const data = await api("/api/routines/run", { method: "POST", body: JSON.stringify({ name, params }) });
    streamJob(data.job_id, {
      onLine: routineLog,
      onDone: () => {
        button.disabled = false;
        showToast(`'${name}' finished.`);
      },
    });
  } catch (e) {
    showToast(`Could not run '${name}': ${e.message}`, true);
    button.disabled = false;
  }
}

// --- chess ---------------------------------------------------------------

let chessSessionId = null;
let chessState = null;
let chessSelectedSquare = null;

const FILES = "abcdefgh";

function fenToGrid(fen) {
  const rows = fen.split(" ")[0].split("/");
  const grid = {}; // "e4" -> piece char
  rows.forEach((row, rankIndexFromTop) => {
    const rank = 8 - rankIndexFromTop;
    let file = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) {
        file += parseInt(ch, 10);
      } else {
        grid[`${FILES[file]}${rank}`] = ch;
        file += 1;
      }
    }
  });
  return grid;
}

const PIECE_GLYPHS = {
  P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕", K: "♔",
  p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚",
};

function renderChessBoard() {
  const board = $("#chessboard");
  board.innerHTML = "";
  if (!chessState) return;
  const grid = fenToGrid(chessState.fen);
  const whiteAtBottom = chessState.human_is_white;
  const ranks = whiteAtBottom ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
  const files = whiteAtBottom ? [0, 1, 2, 3, 4, 5, 6, 7] : [7, 6, 5, 4, 3, 2, 1, 0];

  for (const rank of ranks) {
    for (const fileIdx of files) {
      const square = `${FILES[fileIdx]}${rank}`;
      const isLight = (fileIdx + rank) % 2 === 1;
      const cell = document.createElement("div");
      cell.className = `chess-square ${isLight ? "light" : "dark"}`;
      cell.dataset.square = square;
      const piece = grid[square];
      if (piece) cell.textContent = PIECE_GLYPHS[piece] || piece;
      if (square === chessSelectedSquare) cell.classList.add("selected");
      cell.addEventListener("click", () => onChessSquareClick(square, piece));
      board.appendChild(cell);
    }
  }
}

function isHumanTurn() {
  if (!chessState) return false;
  return (chessState.turn === "white") === chessState.human_is_white;
}

async function onChessSquareClick(square, piece) {
  if (!chessState || chessState.game_over || !isHumanTurn()) return;

  const isOwnPiece = piece && (chessState.human_is_white ? piece === piece.toUpperCase() : piece === piece.toLowerCase());

  if (!chessSelectedSquare) {
    if (isOwnPiece) {
      chessSelectedSquare = square;
      renderChessBoard();
    }
    return;
  }

  if (chessSelectedSquare === square) {
    chessSelectedSquare = null;
    renderChessBoard();
    return;
  }

  if (isOwnPiece) {
    chessSelectedSquare = square;
    renderChessBoard();
    return;
  }

  const from = chessSelectedSquare;
  chessSelectedSquare = null;
  let uci = `${from}${square}`;
  const movingPiece = fenToGrid(chessState.fen)[from];
  const destRank = square[1];
  if (movingPiece && movingPiece.toLowerCase() === "p" && (destRank === "8" || destRank === "1")) {
    uci += "q"; // auto-queen — the overwhelmingly common choice
  }

  try {
    chessState = await api("/api/chess/move", { method: "POST", body: JSON.stringify({ session_id: chessSessionId, move: uci }) });
    renderChessBoard();
    updateChessStatus();
  } catch (e) {
    showToast(e.message, true);
    renderChessBoard();
  }
}

function updateChessStatus() {
  if (!chessState) return;
  let text = chessState.status || "";
  if (chessState.game_over) {
    text = `Game over (${chessState.result}). ${text}`;
  } else {
    text = `${chessState.turn === "white" ? "White" : "Black"} to move. ${text}`;
  }
  $("#chess-status").textContent = text;
}

async function newChessGame() {
  const color = $("#chess-color").value;
  chessSelectedSquare = null;
  try {
    chessState = await api("/api/chess/new", { method: "POST", body: JSON.stringify({ color }) });
    chessSessionId = chessState.session_id;
    renderChessBoard();
    updateChessStatus();
  } catch (e) {
    showToast(`Could not start game: ${e.message}`, true);
  }
}

$("#chess-new").addEventListener("click", newChessGame);

// --- premium ---------------------------------------------------------------

async function loadPremium() {
  const wallet = await api("/api/premium/wallet");
  $("#wallet-address").textContent = wallet.address;
  $("#wallet-email").textContent = wallet.email;
  $("#wallet-qr-img").src = "/api/premium/qrcode.png";
  await refreshPremiumStatus();
}

async function refreshPremiumStatus() {
  const status = await api("/api/premium/status");
  const badge = $("#premium-badge");
  const statusLine = $("#premium-status");
  if (status.unlocked) {
    badge.innerHTML = '<span class="premium-badge">&#9733; PREMIUM</span>';
    statusLine.textContent = `Premium unlocked for ${status.email}.`;
  } else {
    badge.innerHTML = "";
    statusLine.textContent = "Not unlocked yet.";
  }
}

async function unlockPremium() {
  const email = $("#premium-email").value.trim();
  const code = $("#premium-code").value.trim();
  if (!email || !code) {
    showToast("Enter both your email and your code.", true);
    return;
  }
  try {
    await api("/api/premium/verify", { method: "POST", body: JSON.stringify({ email, code }) });
    showToast("Premium unlocked. Thank you for supporting deskbot!");
    await refreshPremiumStatus();
  } catch (e) {
    showToast(e.message, true);
  }
}

$("#premium-unlock").addEventListener("click", unlockPremium);

// --- boot ------------------------------------------------------------------

loadPersonas().catch(() => {});
loadRoutines().catch(() => {});
loadPremium().catch(() => {});

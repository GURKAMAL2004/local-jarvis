(() => {
  "use strict";

  // ===========================================================================
  // deskbot watch — distraction-free YouTube kiosk, 1-4 pane video wall.
  //
  // No-button rule: there is exactly one control surface in this app — the
  // text composer. Volume, mute, play/pause, closing a pane, changing the
  // split layout, and resizing a pane's share of the screen are ALL typed
  // commands, not clicks, drags, or keyboard shortcuts. Once any pane has a
  // video, every submitted message is first sent to the backend command
  // classifier (/api/watch/command, see classify_command() in
  // deskbot/webui/watch.py), which decides whether it's a new search or a
  // control action, and which pane it's about — using each pane's title,
  // playback state, and which pane was most recently active for context
  // (so "turn it up" / "pause the recipe one" / "make this bigger" all
  // resolve without the user ever naming an index).
  //
  // Each pane still owns an independent backend WatchSession (ask -> search
  // -> play) and an independent YT.Player instance — that part is unchanged
  // from the multi-pane build; only the input surface changed.
  // ===========================================================================

  const REQUEST_TIMEOUT_MS = 60_000;
  const THINKING_HINT_DELAY_MS = 6_000;
  const PANE_COUNT = 4;
  const SPLIT_MIN = 0.15;
  const SPLIT_MAX = 0.85;
  const VOLUME_STEP = 15;
  const RESIZE_STEP = 0.12;

  const thread = document.getElementById("thread");
  const form = document.getElementById("composer-form");
  const input = document.getElementById("composer-input");
  const sendBtn = document.getElementById("composer-send");
  const stageGrid = document.getElementById("stage-grid");

  // --- layout definitions ---------------------------------------------------
  //
  // Grid tracks are [fr][6px splitter][fr] per resizable axis, so column/row
  // line numbers below are always 1 (start), 2 (splitter), 3 (fixed after a
  // splitter), 4 (end).

  const LAYOUTS = {
    1: {
      columns: "1fr",
      rows: "1fr",
      areas: [{ pane: 0, col: "1 / 2", row: "1 / 2" }],
      splitters: [],
    },
    2: {
      columns: "var(--c0, 1fr) 6px var(--c1, 1fr)",
      rows: "1fr",
      areas: [
        { pane: 0, col: "1 / 2", row: "1 / 2" },
        { pane: 1, col: "3 / 4", row: "1 / 2" },
      ],
      splitters: [{ type: "v", col: "2 / 3", row: "1 / 2" }],
    },
    3: {
      columns: "var(--c0, 1fr) 6px var(--c1, 1fr)",
      rows: "var(--r0, 1fr) 6px var(--r1, 1fr)",
      areas: [
        { pane: 0, col: "1 / 2", row: "1 / 4" },
        { pane: 1, col: "3 / 4", row: "1 / 2" },
        { pane: 2, col: "3 / 4", row: "3 / 4" },
      ],
      splitters: [
        { type: "v", col: "2 / 3", row: "1 / 4" },
        { type: "h", col: "3 / 4", row: "2 / 3" },
      ],
    },
    4: {
      columns: "var(--c0, 1fr) 6px var(--c1, 1fr)",
      rows: "var(--r0, 1fr) 6px var(--r1, 1fr)",
      areas: [
        { pane: 0, col: "1 / 2", row: "1 / 2" },
        { pane: 1, col: "3 / 4", row: "1 / 2" },
        { pane: 2, col: "1 / 2", row: "3 / 4" },
        { pane: 3, col: "3 / 4", row: "3 / 4" },
      ],
      // Two horizontal segments (left + right column), both driven by the
      // same --r0/--r1 vars so they always move together as one line —
      // avoids one splitter having to span across the vertical one.
      splitters: [
        { type: "v", col: "2 / 3", row: "1 / 4" },
        { type: "h", col: "1 / 2", row: "2 / 3" },
        { type: "h", col: "3 / 4", row: "2 / 3" },
      ],
    },
  };

  // How each pane's "bigger"/"smaller" maps to the shared --c/--r ratio.
  // Each entry is [axis, sign] — sign is which direction on that axis grows
  // *this* pane.
  const RESIZE_MAP = {
    1: {},
    2: { 0: [["v", 1]], 1: [["v", -1]] },
    3: { 0: [["v", 1]], 1: [["v", -1], ["h", 1]], 2: [["v", -1], ["h", -1]] },
    4: {
      0: [["v", 1], ["h", 1]],
      1: [["v", -1], ["h", 1]],
      2: [["v", 1], ["h", -1]],
      3: [["v", -1], ["h", -1]],
    },
  };

  let currentLayout = 1;
  let lastActivePane = null;
  let activeConversationPaneIndex = null;
  let busy = false;
  let splitRatios = { v: 0.5, h: 0.5 };

  // --- pane construction -----------------------------------------------------

  const panes = [];

  function buildPanes() {
    for (let i = 0; i < PANE_COUNT; i++) {
      const el = document.createElement("div");
      el.className = "pane";
      el.hidden = true;
      el.style.setProperty("--pane-color", `var(--pane-color-${i})`);

      const emptyEl = document.createElement("div");
      emptyEl.className = "pane-empty";
      const emptyIcon = document.createElement("div");
      emptyIcon.className = "pane-empty-icon";
      emptyIcon.textContent = "▶";
      const emptyText = document.createElement("div");
      emptyText.className = "pane-empty-text";
      emptyText.textContent = "What do you want to watch?";
      emptyEl.append(emptyIcon, emptyText);

      const playerWrap = document.createElement("div");
      playerWrap.className = "pane-player-wrap";
      playerWrap.hidden = true;
      const mount = document.createElement("div");
      mount.className = "pane-mount";
      playerWrap.appendChild(mount);

      const overlay = document.createElement("div");
      overlay.className = "pane-overlay";
      const title = document.createElement("div");
      title.className = "pane-title";
      overlay.appendChild(title);

      el.append(emptyEl, playerWrap, overlay);
      stageGrid.appendChild(el);

      panes.push({
        index: i,
        el, emptyEl, playerWrap, mount, title,
        player: null,
        hasVideo: false,
        sessionId: null,
        candidates: [],
        candidateIndex: 0,
        announcedIndex: -1,
        isMuted: false,
      });
    }
  }

  function setLastActivePane(i) {
    lastActivePane = i;
    panes.forEach((p) => p.el.classList.toggle("focused", p.index === i));
  }

  // --- pane control primitives (called only from dispatched LLM actions) -----

  function togglePlayback(pane, command) {
    if (!pane.player || !pane.hasVideo) return;
    const state = pane.player.getPlayerState();
    if (command === "pause" && state === YT.PlayerState.PLAYING) pane.player.pauseVideo();
    else if (command === "play" && state !== YT.PlayerState.PLAYING) pane.player.playVideo();
  }

  function setMuted(pane, muted) {
    if (!pane.player) return;
    if (muted && !pane.player.isMuted()) {
      pane.player.mute();
      pane.isMuted = true;
    } else if (!muted && pane.player.isMuted()) {
      pane.player.unMute();
      pane.isMuted = false;
    }
  }

  function adjustVolume(pane, direction) {
    if (!pane.player) return;
    const delta = direction === "up" ? VOLUME_STEP : -VOLUME_STEP;
    const next = Math.min(100, Math.max(0, pane.player.getVolume() + delta));
    pane.player.setVolume(next);
    if (next > 0 && pane.isMuted) setMuted(pane, false);
  }

  function pausePane(pane) {
    if (pane.player && pane.hasVideo) {
      try { pane.player.pauseVideo(); } catch (e) { /* not ready yet */ }
    }
  }

  function closePane(pane) {
    if (pane.player) {
      try { pane.player.stopVideo(); } catch (e) { /* already stopped/never started */ }
    }
    pane.hasVideo = false;
    pane.sessionId = null;
    pane.candidates = [];
    pane.candidateIndex = 0;
    pane.announcedIndex = -1;
    pane.title.textContent = "";
    pane.playerWrap.hidden = true;
    pane.playerWrap.classList.remove("show");
    pane.emptyEl.hidden = false;
    if (activeConversationPaneIndex === pane.index) activeConversationPaneIndex = null;
    updateGlobalPlayingState();
  }

  function resizePane(paneIndex, direction) {
    const rules = (RESIZE_MAP[currentLayout] || {})[paneIndex];
    if (!rules || !rules.length) return;
    const sign = direction === "bigger" ? 1 : -1;
    rules.forEach(([axis, ruleSign]) => {
      splitRatios[axis] = Math.min(SPLIT_MAX, Math.max(SPLIT_MIN, splitRatios[axis] + RESIZE_STEP * sign * ruleSign));
    });
    applySplitRatios();
  }

  function applySplitRatios() {
    stageGrid.style.setProperty("--c0", splitRatios.v.toFixed(4) + "fr");
    stageGrid.style.setProperty("--c1", (1 - splitRatios.v).toFixed(4) + "fr");
    stageGrid.style.setProperty("--r0", splitRatios.h.toFixed(4) + "fr");
    stageGrid.style.setProperty("--r1", (1 - splitRatios.h).toFixed(4) + "fr");
  }

  // --- layout switching --------------------------------------------------------

  function applyLayout(n) {
    const config = LAYOUTS[n];
    if (!config) return;
    currentLayout = n;
    splitRatios = { v: 0.5, h: 0.5 };
    applySplitRatios();

    stageGrid.style.gridTemplateColumns = config.columns;
    stageGrid.style.gridTemplateRows = config.rows;
    stageGrid.dataset.layout = String(n);

    panes.forEach((pane) => {
      const area = config.areas.find((a) => a.pane === pane.index);
      if (area) {
        pane.el.hidden = false;
        pane.el.style.gridColumn = area.col;
        pane.el.style.gridRow = area.row;
        pane.el.dataset.layoutSize = String(n);
      } else {
        pane.el.hidden = true;
        pausePane(pane); // don't leave audio playing in a pane you can't see
      }
    });

    stageGrid.querySelectorAll(".splitter").forEach((el) => el.remove());
    config.splitters.forEach((s) => {
      const el = document.createElement("div");
      el.className = "splitter splitter-" + s.type;
      el.style.gridColumn = s.col;
      el.style.gridRow = s.row;
      stageGrid.appendChild(el);
    });

    if (lastActivePane !== null && lastActivePane >= n) setLastActivePane(null);
  }

  // --- chat plumbing -----------------------------------------------------------

  function addBubble(text, role, paneIndex) {
    const el = document.createElement("div");
    el.className = "bubble " + role;
    if (typeof paneIndex === "number" && currentLayout > 1) {
      const dot = document.createElement("span");
      dot.className = "bubble-pane-dot";
      dot.style.background = `var(--pane-color-${paneIndex})`;
      el.appendChild(dot);
    }
    const span = document.createElement("span");
    span.textContent = text;
    el.appendChild(span);
    thread.prepend(el);
    return el;
  }

  function setBusy(state) {
    busy = state;
    sendBtn.disabled = state;
    input.disabled = state;
  }

  let thinkingBubble = null;
  let thinkingHintTimer = null;

  function showThinking(paneIndex) {
    thinkingBubble = document.createElement("div");
    thinkingBubble.className = "bubble thinking";
    if (currentLayout > 1 && typeof paneIndex === "number") {
      const dot = document.createElement("span");
      dot.className = "bubble-pane-dot";
      dot.style.background = `var(--pane-color-${paneIndex})`;
      thinkingBubble.appendChild(dot);
    }
    const dots = document.createElement("span");
    dots.className = "dots";
    dots.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
    thinkingBubble.appendChild(dots);
    thread.prepend(thinkingBubble);
    thinkingHintTimer = setTimeout(() => {
      addBubble("Still thinking — local models can take a moment.", "hint", paneIndex);
    }, THINKING_HINT_DELAY_MS);
  }

  function hideThinking() {
    clearTimeout(thinkingHintTimer);
    if (thinkingBubble) {
      thinkingBubble.remove();
      thinkingBubble = null;
    }
    thread.querySelectorAll(".bubble.hint").forEach((el) => el.remove());
  }

  function updateGlobalPlayingState() {
    const anyPlaying = panes.some((p) => p.hasVideo);
    document.body.classList.toggle("playing", anyPlaying);
    input.placeholder = anyPlaying ? "Play something else, or tell it what to do…" : "What do you want to watch?";
  }

  function pickTargetPaneIndex() {
    for (let i = 0; i < currentLayout; i++) {
      if (!panes[i].hasVideo) return i;
    }
    if (lastActivePane !== null && !panes[lastActivePane].el.hidden) return lastActivePane;
    return 0;
  }

  function clampPaneIndex(index) {
    if (typeof index !== "number" || index < 0 || index >= currentLayout) return pickTargetPaneIndex();
    return index;
  }

  function resolvePane(index) {
    if (typeof index !== "number") return null;
    if (index < 0 || index >= panes.length || panes[index].el.hidden) return null;
    return panes[index];
  }

  // --- deterministic fast paths --------------------------------------------
  //
  // Layout changes and short, unambiguous control phrases never depend on
  // the local model — verified against a real local model that even once
  // reachable reliably garbled the "how many panes" JSON (the word "panes"
  // collides with the per-pane state list already in the same prompt).
  // Regex beats a coin-flip here. Anything these don't confidently
  // recognize still falls through to the LLM classifier for real language
  // understanding (title matches, "the recipe video", etc).

  const NUMBER_WORDS = { one: 1, two: 2, three: 3, four: 4 };

  function toLayoutNumber(raw) {
    const n = NUMBER_WORDS[raw] !== undefined ? NUMBER_WORDS[raw] : parseInt(raw, 10);
    return n >= 1 && n <= 4 ? n : null;
  }

  function matchLayoutCommand(text) {
    const t = text.toLowerCase().trim();
    const num = "(1|2|3|4|one|two|three|four)";

    // Non-capturing groups everywhere except the number itself, so match[1]
    // is unambiguously the number regardless of which side of the noun it
    // falls on ("4 screens" vs "split into 4") — no guessing which capture
    // group is which.
    const numberThenNoun = t.match(new RegExp(`\\b${num}\\b[\\s-]*(?:way|videos?|screens?|panes?|windows?)\\b`));
    if (numberThenNoun) {
      const n = toLayoutNumber(numberThenNoun[1]);
      if (n) return n;
    }

    const nounThenNumber = t.match(new RegExp(`\\b(?:split(?:\\s*screen)?|screens?|panes?)\\b.*\\b${num}\\b`));
    if (nounThenNumber) {
      const n = toLayoutNumber(nounThenNumber[1]);
      if (n) return n;
    }

    if (/\b(back to one|single (video|screen)|one video only|full ?screen|just one video)\b/.test(t)) return 1;
    if (/\bsplit\s*screen\b/.test(t) && !/[1-4]|one|two|three|four/.test(t)) return 2;

    return null;
  }

  function matchSimpleControlCommand(text) {
    const t = text.toLowerCase().trim();
    if (/^(turn it up|louder|volume up|turn up the volume|increase( the)? volume)$/.test(t)) return { action: "volume", direction: "up" };
    if (/^(turn it down|quieter|volume down|turn down the volume|lower( the)? volume|decrease( the)? volume)$/.test(t)) return { action: "volume", direction: "down" };
    if (/^(mute( it)?|mute the video)$/.test(t)) return { action: "mute" };
    if (/^(unmute( it)?|unmute the video)$/.test(t)) return { action: "unmute" };
    if (/^(pause( it)?|pause the video)$/.test(t)) return { action: "playback", command: "pause" };
    if (/^(play( it)?( again)?|resume( it)?)$/.test(t)) return { action: "playback", command: "play" };
    if (/^(close it|close the video|stop it|stop the video)$/.test(t)) return { action: "close" };
    if (/^(make (it|this) bigger|bigger)$/.test(t)) return { action: "resize", direction: "bigger" };
    if (/^(make (it|this) smaller|smaller)$/.test(t)) return { action: "resize", direction: "smaller" };
    return null;
  }

  function defaultTargetPaneForControl() {
    if (lastActivePane !== null && panes[lastActivePane].hasVideo) return lastActivePane;
    const withVideo = panes.filter((p) => p.hasVideo);
    return withVideo.length === 1 ? withVideo[0].index : null;
  }

  // Verb + a title reference ("mute the lofi one", "close the recipe video")
  // — the exact case that depended on the LLM before and wasn't reliable at
  // it. Only ever fires when the descriptor after the verb matches exactly
  // one pane's title; otherwise it backs off to null rather than guess, and
  // the message falls through to the LLM classifier as before.
  const STOPWORDS = new Set(["that", "this", "one", "video", "screen", "pane", "the", "volume", "it"]);

  const CONTROL_VERB_PATTERNS = [
    { re: /^mute\s+(?:the\s+|that\s+)?(.+)$/, action: () => ({ action: "mute" }) },
    { re: /^unmute\s+(?:the\s+|that\s+)?(.+)$/, action: () => ({ action: "unmute" }) },
    { re: /^(?:pause|stop)\s+(?:the\s+|that\s+)?(.+)$/, action: () => ({ action: "playback", command: "pause" }) },
    { re: /^close\s+(?:the\s+|that\s+)?(.+)$/, action: () => ({ action: "close" }) },
    { re: /^turn\s+up\s+(?:the\s+volume\s+(?:on|for)\s+)?(?:the\s+|that\s+)?(.+)$/, action: () => ({ action: "volume", direction: "up" }) },
    { re: /^turn\s+down\s+(?:the\s+volume\s+(?:on|for)\s+)?(?:the\s+|that\s+)?(.+)$/, action: () => ({ action: "volume", direction: "down" }) },
    { re: /^make\s+(?:the\s+|that\s+)?(.+?)\s+bigger$/, action: () => ({ action: "resize", direction: "bigger" }) },
    { re: /^make\s+(?:the\s+|that\s+)?(.+?)\s+smaller$/, action: () => ({ action: "resize", direction: "smaller" }) },
  ];

  function findPaneByTitleWords(descriptor) {
    const words = descriptor
      .toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/)
      .filter((w) => w.length >= 4 && !STOPWORDS.has(w));
    if (!words.length) return null;

    const matches = panes.filter((p) => {
      if (!p.hasVideo) return false;
      const title = p.title.textContent.toLowerCase();
      return words.some((w) => title.includes(w));
    });
    return matches.length === 1 ? matches[0].index : null;
  }

  function matchControlVerbWithTarget(text) {
    const t = text.toLowerCase().trim();
    for (const { re, action } of CONTROL_VERB_PATTERNS) {
      const m = t.match(re);
      if (!m) continue;
      const descriptor = (m[1] || "").trim();
      if (!descriptor) continue; // e.g. "mute" alone — matchSimpleControlCommand already owns that
      const paneIndex = findPaneByTitleWords(descriptor);
      if (paneIndex !== null) return { ...action(), pane: paneIndex };
    }
    return null;
  }

  // --- top-level message routing ------------------------------------------

  async function send(message) {
    // Continuing an unanswered clarifying question always wins — never
    // reinterpret a reply like "the iPhone 12" as a volume command.
    if (activeConversationPaneIndex !== null) {
      addBubble(message, "user", activeConversationPaneIndex);
      await runSearchTurn(message, activeConversationPaneIndex);
      return;
    }

    // Layout is recognized regardless of whether anything is playing yet —
    // "split into 4" typed before any search should set up the screen, not
    // get treated as a video to search for.
    const layoutCount = matchLayoutCommand(message);
    if (layoutCount !== null) {
      addBubble(message, "user", lastActivePane);
      applyLayout(layoutCount);
      return;
    }

    const anyPlaying = panes.some((p) => p.hasVideo);

    if (anyPlaying) {
      const simple = matchSimpleControlCommand(message);
      if (simple) {
        const target = defaultTargetPaneForControl();
        if (target !== null) {
          addBubble(message, "user", target);
          applyControlAction({ ...simple, pane: target });
          return;
        }
        // Ambiguous which pane (multiple videos, none recently active) —
        // fall through below rather than guess.
      } else {
        const verbMatch = matchControlVerbWithTarget(message);
        if (verbMatch) {
          addBubble(message, "user", verbMatch.pane);
          applyControlAction(verbMatch);
          return;
        }
      }
    }

    if (!anyPlaying) {
      const target = pickTargetPaneIndex();
      addBubble(message, "user", target);
      await runSearchTurn(message, target);
      return;
    }

    addBubble(message, "user", lastActivePane);
    await routeThroughCommandClassifier(message);
  }

  async function withTimeout(fetchPromiseFactory, onTimeoutBubblePane) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      return await fetchPromiseFactory(controller.signal);
    } catch (err) {
      if (err.name === "AbortError") {
        addBubble("That took too long and timed out — try again.", "error", onTimeoutBubblePane);
      } else {
        addBubble("Something went wrong — try again.", "error", onTimeoutBubblePane);
      }
      return null;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async function runSearchTurn(message, paneIndex) {
    const pane = panes[paneIndex];
    lastActivePane = paneIndex;
    activeConversationPaneIndex = paneIndex;
    setBusy(true);
    if (!pane.hasVideo) pane.emptyEl.classList.add("busy");
    showThinking(paneIndex);

    const data = await withTimeout(async (signal) => {
      const url = pane.sessionId ? "/api/watch/message" : "/api/watch/start";
      const body = pane.sessionId ? { session_id: pane.sessionId, message } : { request: message };
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      });
      if (!res.ok) throw new Error("request failed: " + res.status);
      return res.json();
    }, paneIndex);

    hideThinking();
    setBusy(false);
    pane.emptyEl.classList.remove("busy");
    input.focus();

    if (data === null) {
      activeConversationPaneIndex = null;
      return;
    }
    if (data.session_id) pane.sessionId = data.session_id;
    handleSearchResponse(data, paneIndex);
  }

  function handleSearchResponse(data, paneIndex) {
    if (data.type === "question") {
      addBubble(data.text, "assistant", paneIndex);
      return; // stays the active conversation pane until it resolves
    }
    activeConversationPaneIndex = null;
    panes[paneIndex].sessionId = null;
    if (data.type === "playing") {
      playInPane(paneIndex, data.candidates && data.candidates.length ? data.candidates : [
        { video_id: data.video_id, title: data.title, channel: data.channel },
      ]);
    } else if (data.type === "error") {
      addBubble(data.text, "error", paneIndex);
    } else {
      addBubble("Unexpected response — try again.", "error", paneIndex);
    }
  }

  async function routeThroughCommandClassifier(message) {
    setBusy(true);
    showThinking(lastActivePane);

    const data = await withTimeout(async (signal) => {
      const res = await fetch("/api/watch/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: message,
          panes: panes.map((p) => ({
            index: p.index,
            title: p.title.textContent,
            has_video: p.hasVideo,
            playing: !!(p.player && p.hasVideo && p.player.getPlayerState() === YT.PlayerState.PLAYING),
            muted: p.isMuted,
            volume: p.player ? p.player.getVolume() : 100,
          })),
          last_active_pane: lastActivePane,
          layout: currentLayout,
        }),
        signal,
      });
      if (!res.ok) throw new Error("request failed: " + res.status);
      return res.json();
    }, lastActivePane);

    hideThinking();
    setBusy(false);
    input.focus();

    if (data === null) return;
    dispatchActions(data.actions || [], message);
  }

  function dispatchActions(actions, message) {
    if (!actions.length) {
      addBubble("Not sure what you meant — try again.", "error", lastActivePane);
      return;
    }

    let handledAny = false;
    let searchAction = null;

    for (const action of actions) {
      if (action.action === "search") {
        searchAction = searchAction || action; // only one search makes sense per message
        continue;
      }
      if (applyControlAction(action)) handledAny = true;
    }

    if (searchAction) {
      const target = searchAction.pane === "new" || searchAction.pane === undefined
        ? pickTargetPaneIndex()
        : clampPaneIndex(searchAction.pane);
      runSearchTurn(message, target);
      return; // runSearchTurn owns setBusy/input.focus from here
    }

    if (!handledAny) {
      addBubble("Not sure what you meant — try again.", "error", lastActivePane);
    }
  }

  function applyControlAction(action) {
    switch (action.action) {
      case "volume": {
        const pane = resolvePane(action.pane);
        if (!pane || !pane.hasVideo) return false;
        adjustVolume(pane, action.direction === "down" ? "down" : "up");
        setLastActivePane(pane.index);
        return true;
      }
      case "mute": {
        const pane = resolvePane(action.pane);
        if (!pane || !pane.hasVideo) return false;
        setMuted(pane, true);
        setLastActivePane(pane.index);
        return true;
      }
      case "unmute": {
        const pane = resolvePane(action.pane);
        if (!pane || !pane.hasVideo) return false;
        setMuted(pane, false);
        setLastActivePane(pane.index);
        return true;
      }
      case "playback": {
        const pane = resolvePane(action.pane);
        if (!pane || !pane.hasVideo) return false;
        togglePlayback(pane, action.command === "play" ? "play" : "pause");
        setLastActivePane(pane.index);
        return true;
      }
      case "close": {
        const pane = resolvePane(action.pane);
        if (!pane || !pane.hasVideo) return false;
        closePane(pane);
        return true;
      }
      case "layout": {
        const n = Number(action.screen_count);
        if (n >= 1 && n <= 4) { applyLayout(n); return true; }
        return false;
      }
      case "resize": {
        const pane = resolvePane(action.pane);
        if (!pane) return false;
        resizePane(pane.index, action.direction === "smaller" ? "smaller" : "bigger");
        setLastActivePane(pane.index);
        return true;
      }
      default:
        return false;
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const value = input.value.trim();
    if (!value || busy) return;
    input.value = "";
    send(value);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      input.value = "";
      input.blur();
    }
  });

  // --- YouTube playback ----------------------------------------------------
  //
  // Deliberately uses the official IFrame Player API (not a bare
  // <iframe src="...">): a plain iframe gives no way to detect "this video
  // can't be embedded" (a common case for official/label music videos) —
  // it just shows YouTube's own broken-looking "Video unavailable / Watch
  // on YouTube" screen. The Player API's onError callback lets us catch
  // that and silently fall through to the next real result already found.

  let apiReady = false;
  let pendingPlays = [];

  function loadYouTubeApi() {
    const tag = document.createElement("script");
    tag.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(tag);
  }

  window.onYouTubeIframeAPIReady = () => {
    apiReady = true;
    const toPlay = pendingPlays;
    pendingPlays = [];
    toPlay.forEach((i) => attemptCurrentCandidate(i));
  };

  function playInPane(paneIndex, candidateList) {
    const pane = panes[paneIndex];
    pane.emptyEl.hidden = true;
    pane.playerWrap.hidden = false;
    requestAnimationFrame(() => pane.playerWrap.classList.add("show"));
    pane.hasVideo = true;
    pane.candidates = candidateList;
    pane.candidateIndex = 0;
    pane.announcedIndex = -1;
    setLastActivePane(paneIndex);
    updateGlobalPlayingState();

    if (!apiReady) {
      pendingPlays.push(paneIndex);
      return;
    }
    attemptCurrentCandidate(paneIndex);
  }

  function attemptCurrentCandidate(paneIndex) {
    const pane = panes[paneIndex];
    if (pane.candidateIndex >= pane.candidates.length) {
      addBubble("None of the matches for that could play here — try rephrasing what you want.", "error", paneIndex);
      return;
    }
    const videoId = pane.candidates[pane.candidateIndex].video_id;
    if (pane.player) {
      pane.player.loadVideoById(videoId);
    } else {
      pane.player = new YT.Player(pane.mount, {
        height: "100%",
        width: "100%",
        videoId,
        playerVars: { autoplay: 1, fs: 1, rel: 0, modestbranding: 1, playsinline: 1 },
        events: {
          onStateChange: (e) => onPlayerStateChange(paneIndex, e),
          onError: () => onPlayerError(paneIndex),
        },
      });
    }
  }

  function onPlayerStateChange(paneIndex, event) {
    const pane = panes[paneIndex];
    if (event.data === YT.PlayerState.PLAYING && pane.announcedIndex !== pane.candidateIndex) {
      pane.announcedIndex = pane.candidateIndex;
      const current = pane.candidates[pane.candidateIndex];
      const label = current.title + (current.channel ? " — " + current.channel : "");
      pane.title.textContent = label;
      addBubble("Playing: " + label, "assistant", paneIndex);
    }
  }

  function onPlayerError(paneIndex) {
    // Error codes 2/5/100/101/150 all mean "this specific video won't play
    // here" (bad id, disallowed embedding, removed, etc.) — none are worth
    // surfacing; just try the next real result already found for this pane.
    const pane = panes[paneIndex];
    pane.candidateIndex += 1;
    attemptCurrentCandidate(paneIndex);
  }

  // --- init --------------------------------------------------------------

  buildPanes();
  applyLayout(1);
  loadYouTubeApi();
})();

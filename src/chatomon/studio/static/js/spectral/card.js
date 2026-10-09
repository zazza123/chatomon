import { $, esc, fmtMs, rich } from "../util.js";
import { send } from "../api.js";
import { openDialog } from "../cards.js";
import { ghostIcon, ghostLive } from "./ghost.js";

/* The chip in the header and the Spectral card. The server says what the connection is doing,
   in its own words; this file decides how that looks and asks again while it can change. */

const WARN_OFF = "While connected, Spectral can send messages to this atom. Each request is a real turn: it calls your models and runs the atom's tools, as many times as the evaluation asks.";
const WARN_ON = "Spectral can reach this atom until you disconnect or stop the studio. Each request is a real turn, with the orbitals as they are when it arrives. Replies go back as text; tool calls are not reported.";
const NOT_LOCAL = "Spectral can be connected only when the studio listens on this machine.";
const FINAL = ["answered", "guided_refusal", "orbital_response", "blocked", "fallback_refusal"];
const REFUSED = { key: "key refused", protocol: "upgrade needed" };

/** How a state of the connection looks: the ghost's pose, the colour of the chip and its last words. */
export function look(state) {
  switch (state.status) {
    case "connecting": return { pose: "connect", tone: "on", tail: "connecting" };
    case "closing": return { pose: "connect", tone: "on", tail: "disconnecting" };
    case "connected": return state.active ? { pose: "evaluate", tone: "on", tail: "requests" } : { pose: "orbit", tone: "on", tail: "connected" };
    case "lost": return { pose: "lost", tone: "warn", tail: "reconnecting" };
    case "refused": return { pose: "refused", tone: "block", tail: REFUSED[state.reason] || "stopped" };
    default: return { pose: "sleep", tone: "", tail: "" };
  }
}

// A count is drawn as a number, whatever arrived in its place.
const num = (value) => Number(value) || 0;
const many = (n, word) => `${num(n)} ${word}${num(n) === 1 ? "" : "s"}`;
const held = (state) => ["connecting", "connected", "lost", "closing"].includes(state.status);
/** Only an address the server gave, and only one a browser opens as a page. */
const link = (site) => (typeof site === "string" && site.startsWith("https://") ? esc(site) : "");

function statsMarkup(counts) {
  const finals = FINAL.filter((name) => num(counts.final[name]) > 0).map((name) => `<span class="pill fa-${name}">${name} ${num(counts.final[name])}</span>`).join("");
  const cell = (n) => `<td${num(n) ? "" : ' class="zero"'}>${num(n)}</td>`;
  const bar = (row) => [["a", row.allow], ["r", row.respond], ["b", row.block], ["e", row.error], ["s", row.not_run]]
    .filter(([, n]) => num(n) > 0).map(([name, n]) => `<i class="${name}" style="flex:${num(n)}"></i>`).join("");
  const rows = counts.orbitals.map((row) =>
    `<tr><th scope="row"><b>${esc(row.name)}</b><i>${esc(row.side)} · ${esc(row.kind)}</i><div class="os-bar" aria-hidden="true">${bar(row)}</div></th>${cell(row.allow)}${cell(row.block)}${cell(row.respond)}${cell(row.error)}${cell(row.not_run)}<td>${num(row.runs) ? fmtMs(num(row.duration_ms_total) / num(row.runs)) : "—"}</td></tr>`).join("");
  const table = rows
    ? `<div class="os-scroll"><table class="os"><thead><tr><th scope="col">orbital</th><th scope="col">allow</th><th scope="col">block</th><th scope="col">respond</th><th scope="col">error</th><th scope="col">not run</th><th scope="col">average</th></tr></thead><tbody>${rows}</tbody></table></div>`
    : "";
  const requests = num(counts.requests);
  const failed = num(counts.failed) ? `<div class="bad"><b>${num(counts.failed)}</b><span>failed</span></div>` : "";
  const changes = num(counts.configurations) - 1;
  const mixed = changes > 0
    ? `<p class="note warn">The orbitals changed ${changes === 1 ? "once" : `${changes} times`} while these counts were running, so the numbers mix more than one configuration. Reset the counts to start clean.</p>`
    : "";
  const rejected = num(counts.rejected)
    ? `<p class="note warn">${many(counts.rejected, "request")} got no turn: Spectral sent something this studio does not read, or continued a conversation that is no longer kept.</p>`
    : "";
  return `<div class="stat-nums"><div><b>${requests}</b><span>requests</span></div><div><b>${num(counts.running)}</b><span>running now</span></div><div><b>${requests ? fmtMs(num(counts.duration_ms_total) / requests) : "—"}</b><span>average turn</span></div>${failed}</div>
    ${finals ? `<div class="finals">${finals}</div>` : `<p class="muted">No request yet.</p>`}${table}${mixed}${rejected}`;
}

const keptText = (n) => (num(n)
  ? `Spectral continues a conversation by naming the answer before it: ${many(n, "conversation")} ${n === 1 ? "is" : "are"} held in memory for that until you disconnect, never shown and never written.`
  : "");

export function createCard({ chip, live, ghost, stage, toast }) {
  let state = null;
  let open = false;
  // What the developer wrote as the name of the variable: kept here, the server keeps only a name it found set.
  let typed = null;
  let problem = "";
  let drawn = "";
  let chipDrawn = "";
  let said = "";
  let resting = false;
  let timer = 0;

  /** What decides the parts of the card: when it changes, the card is drawn again. */
  const shape = () => JSON.stringify([
    state.status, state.reason, state.local, state.api_key_env, state.key_set, state.adapter, state.test,
    state.notes, state.counts === null, state.site,
  ]);

  function syncChip() {
    const { pose, tone, tail } = look(state);
    const requests = state.counts ? num(state.counts.requests) : 0;
    const signature = `${pose}:${tone}:${tail}`;
    if (signature !== chipDrawn) {
      chipDrawn = signature;
      const words = tail === "requests" ? `<span class="n">${requests}</span> <span class="unit"></span>` : tail;
      chip.dataset.tone = tone;
      chip.innerHTML = `${ghostIcon(pose)}<span class="txt">Spectral${words ? `<span class="long"> · ${words}</span>` : ""}</span>`;
    }
    const n = $(".n", chip);
    if (n) {
      n.textContent = requests;
      $(".unit", chip).textContent = requests === 1 ? "request" : "requests";
    }
    chip.title = state.message;
    chip.setAttribute("aria-label", `${state.message}. Open the Spectral card`);
  }

  function syncStage() {
    const { pose } = look(state);
    ghost.set(pose);
    // With requests arriving and no turn of the developer's, the stage says what the atom is doing.
    const busy = pose === "evaluate";
    if (busy !== resting) {
      resting = busy;
      stage.setRest(busy ? ["Answering Spectral", "Its requests are counted, not shown here"] : null);
    }
    ghost.settle();
  }

  /** What the relay row says: the server's sentence, and since when the connection is open. */
  function relayText() {
    const since = state.since ? ` · since ${new Date(state.since).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` : "";
    return `${state.message}${since}`;
  }

  function checks() {
    if (!state.local) return `<p class="note warn">${NOT_LOCAL}</p>`;
    const locked = held(state);
    const on = state.adapter !== null;
    const name = typed ?? state.api_key_env;
    const known = name.trim() === state.api_key_env;
    const pill = `<span class="pill ${state.key_set ? "act-allow" : "act-block"}" id="sp-env-pill"${known ? "" : " hidden"}>${state.key_set ? "set" : "not set"}</span>`;
    const bad = state.status === "refused" || state.status === "lost";
    const routes = on ? state.adapter.routes.map((route) => `<span class="mono">${esc(route)}</span>`).join(" and ") : "";
    const test = state.test;
    const tested = !test ? "Not tested yet"
      : test.outcome === "running" ? "Running one turn…"
      : test.outcome === "failed" ? esc(test.message)
      : `<span class="pill fa-${esc(test.final_action)}">${esc(test.final_action)}</span><span class="mono">${fmtMs(num(test.duration_ms))}</span>`;
    return `<div class="checks">
      <div class="check"><label class="label" for="sp-env">Key</label>
        <span class="what"><input id="sp-env" class="mono" type="text" value="${esc(name)}" autocapitalize="none" autocomplete="off" spellcheck="false" aria-describedby="sp-env-help sp-error"${locked ? " disabled" : ""}>${pill}</span><span></span>
        <span class="help" id="sp-env-help">The name of an environment variable. Its value stays on the server.${locked ? " Disconnect to change it." : ""}</span></div>
      <div class="check"><span class="label">Relay</span>
        <span class="what"><span class="mono">${esc(state.relay)}</span><span class="probe${bad ? " err" : ""}" id="sp-relay">${esc(relayText())}</span></span><span></span>
        <span class="help">The address comes from where the studio was started, never from this page.</span></div>
      <div class="check"><span class="label">Adapter</span>
        <span class="what">${on ? `<span class="mono">127.0.0.1:${Number(state.adapter.port)}</span><span class="probe">Open while connected · answers ${routes}</span>` : `<span class="probe">Closed. It opens when you connect.</span>`}</span><span></span>
        <span class="help">A small server that only this machine can reach: spectral-bridge hands it each request. Its address also carries a key of this connection, which is never shown.</span></div>
      <div class="check"><span class="label">Atom</span>
        <span class="what"><span class="probe${test && test.outcome === "failed" ? " err" : ""}" role="status">${tested}</span></span><button class="btn sm" type="button" data-sp="test"${test && test.outcome === "running" ? " disabled" : ""}>Send a test request</button>
        <span class="help">One real turn on a sample message, built like a request from Spectral. It is not added to the conversation and not counted.</span></div>
    </div>`;
  }

  function body() {
    const on = state.adapter !== null;
    const connected = state.status === "connected";
    const notes = !held(state) && state.local && state.notes.length
      ? `<div class="note"><p>About this atom:</p><ul>${state.notes.map((note) => `<li>${rich(note)}</li>`).join("")}</ul></div>`
      : "";
    const notice = !state.local ? "" : `<p class="note${on ? "" : " warn"}">${on ? WARN_ON : WARN_OFF}</p>`;
    const next = `<p class="note" id="sp-next"${connected && state.counts && !state.counts.requests ? "" : " hidden"}>Next: in Spectral, open your target, press <b>Test connection</b>, then launch an evaluation.</p>`;
    const stats = state.counts
      ? `<section class="stats"><div class="stats-head"><h3 class="label">Requests from Spectral</h3><button class="tbtn" type="button" data-sp="reset">Reset the counts</button></div>
          <div class="stats" id="sp-stats">${statsMarkup(state.counts)}</div>
          <p class="foot">Counts only: these requests are not shown, and they are not in the conversation below.</p>
          <p class="foot" id="sp-kept"${state.conversations ? "" : " hidden"}>${keptText(state.conversations)}</p></section>`
      : "";
    return `${checks()}<p class="field-err" id="sp-error" role="alert"${problem ? "" : " hidden"}>${rich(problem)}</p>${notice}${notes}${next}${stats}`;
  }

  function actions() {
    const site = link(state.site);
    const open_ = site ? `<a class="btn" href="${site}" target="_blank" rel="noopener noreferrer">Open Spectral</a>` : "";
    const main = !state.local ? ""
      : held(state) ? `<button class="btn danger" type="button" data-sp="disconnect"${state.status === "closing" ? " disabled" : ""}>${state.status === "connecting" ? "Cancel" : "Disconnect"}</button>`
      : `<button class="btn primary" type="button" data-sp="connect">Connect</button>`;
    return `${main}${open_}<span class="spacer"></span><button class="btn" type="button" data-dlg="close">Done</button>`;
  }

  function show() {
    const again = open;
    // Drawn again, the card keeps its place and gives the focus back to the control that had it.
    const scroll = again ? $("#dlg-body").scrollTop : 0;
    const active = again ? document.activeElement : null;
    const focused = active && (active.id ? `#${active.id}` : active.dataset.sp ? `[data-sp="${active.dataset.sp}"]` : "");
    const site = link(state.site);
    open = true;
    drawn = shape();
    openDialog({
      size: "l",
      lead: ghostLive(look(state).pose, state.message),
      title: "Spectral",
      sub: `Evaluation at scale, by Principled Intelligence${site ? ` · <a href="${site}" target="_blank" rel="noopener noreferrer">${esc(new URL(state.site).host)}</a>` : ""}`,
      body: body(),
      actions: actions(),
      onClose: () => { open = false; problem = ""; plan(); },
    });
    $("#dlg-body").scrollTop = scroll;
    $("#dlg-body").onclick = act;
    $("#dlg-actions").onclick = act;
    $("#dlg-body").oninput = (event) => {
      if (event.target.id !== "sp-env") return;
      typed = event.target.value;
      problem = "";
      $("#sp-error").hidden = true;
      // The mark is about the name the server knows: another name has none until it is tried.
      $("#sp-env-pill").hidden = typed.trim() !== state.api_key_env;
    };
    $("#dlg-body").onkeydown = (event) => {
      if (event.key === "Enter" && event.target.id === "sp-env") connect();
    };
    const target = focused && $(focused, $("#dlg"));
    if (target && !target.disabled) target.focus();
    plan();
  }

  /** What changes while an evaluation runs, shown without drawing the card again. */
  function syncCard() {
    if (!open) return;
    if (shape() !== drawn) { show(); return; }
    const { pose } = look(state);
    const lead = $("#dlg-lead svg");
    if (lead && !lead.classList.contains(`pose-${pose}`)) $("#dlg-lead").innerHTML = ghostLive(pose, state.message);
    const relay = $("#sp-relay");
    if (relay) relay.textContent = relayText();
    if (!state.counts) return;
    $("#sp-stats").innerHTML = statsMarkup(state.counts);
    $("#sp-next").hidden = !(state.status === "connected" && !state.counts.requests);
    $("#sp-kept").hidden = !state.conversations;
    $("#sp-kept").textContent = keptText(state.conversations);
  }

  function apply(next) {
    state = next;
    syncChip();
    syncStage();
    if (state.message !== said) {
      said = state.message;
      live.textContent = said;
    }
    syncCard();
    plan();
  }

  /** Ask again in a second, while something can change and the page is in view. */
  function plan() {
    clearTimeout(timer);
    if (document.hidden || !state || !(open || state.status !== "off")) return;
    timer = setTimeout(async () => {
      // The studio says so itself when it stops answering: here a failure only ends the asking.
      try { apply(await send("GET", "/api/spectral")); } catch { /* nothing to show */ }
    }, 1000);
  }
  document.addEventListener("visibilitychange", plan);

  function refuse(error) {
    problem = error.message;
    if (open) show();
    else toast(error.message);
  }

  async function connect() {
    const name = typed ?? state.api_key_env;
    problem = "";
    try {
      const next = await send("POST", "/api/spectral/connect", { api_key_env: name });
      typed = null;
      apply(next);
    } catch (error) {
      refuse(error);
      $("#sp-env")?.focus();
    }
  }

  async function act(event) {
    const button = event.target.closest("[data-sp]");
    if (!button || button.disabled) return;
    const what = button.dataset.sp;
    if (what === "connect") { await connect(); return; }
    try {
      if (what === "disconnect") apply(await send("POST", "/api/spectral/disconnect", {}));
      if (what === "reset") {
        apply(await send("POST", "/api/spectral/counts/reset", {}));
        toast("The counts start again");
      }
      if (what === "test") {
        button.disabled = true;
        // The answer comes when the turn has ended: meanwhile the card says that one is running.
        plan();
        apply(await send("POST", "/api/spectral/test", {}));
      }
    } catch (error) {
      refuse(error);
    }
  }

  chip.addEventListener("click", () => { if (state) show(); });

  return {
    /** The state of the connection, as the server gives it with the state of the studio. */
    sync: apply,
    /** Open the card, as the chip does. */
    show: () => { if (state) show(); },
  };
}

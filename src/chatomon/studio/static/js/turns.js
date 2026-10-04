import { esc, fmtMs, plural } from "./util.js";

const LENS = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5L21 21"/></svg>`;

/** The results of one side, in chain order; orbitals that did not run get action "skip". */
export function results(turn, side, attempt = -1) {
  const report = turn.report;
  let ran = [];
  if (report && side === "input") ran = report.input;
  if (report && side === "output") ran = report.output.at(attempt) || [];
  const byName = new Map(ran.map((r) => [r.name, r]));
  return turn.chain[side].map((name) => byName.get(name) || { name, action: "skip" });
}

/** Whether the nucleus was called and the call failed before any reply was checked. */
export function nucleusFailed(turn) {
  const report = turn.report;
  const aborted = Boolean(turn.error) || (report && report.final_action === "aborted");
  const attempts = report ? report.output.length : 0;
  const stopped = results(turn, "input").some((x) => x.action === "block" || x.action === "respond");
  return aborted && !attempts && !stopped;
}

export function redacted(turn) {
  return Boolean(turn.report && turn.report.input.some((r) => r.action === "block"));
}

function trail(turn) {
  const cls = (a) => (a === "skip" ? "skip" : a === "block" ? "block" : "");
  const dash = (x) => `<i class="${cls(x.action)}" title="${esc(`${x.name}: ${x.action === "skip" ? "not run" : x.action}`)}"></i>`;
  const attempts = turn.report ? turn.report.output.length : 0;
  const ins = results(turn, "input").map(dash).join("");
  const outs = attempts ? results(turn, "output").map(dash).join("") : "";
  const nucleus = attempts
    ? `<b title="nucleus"></b>`
    : nucleusFailed(turn)
      ? `<b class="fail" title="nucleus failed"></b>`
      : `<b class="skip" title="nucleus not called"></b>`;
  const regen = attempts > 1 ? `<span class="regen" title="${plural(attempts - 1, "regeneration")}">↻${attempts - 1}</span>` : "";
  return `<button class="trail" type="button" data-act="inspect" data-n="${turn.n}" aria-label="Orbital trail of turn ${turn.n}: open the report">${ins}${nucleus}${outs}${regen}</button>`;
}

export function renderTurns(log, turns, { pending, busy, inspected, symbol, placeholder }) {
  if (!turns.length && !pending) {
    log.innerHTML = `<p class="empty">Send a message to the atom. Each turn appears here; open its lens to see which orbitals ran and what they decided.</p>`;
    return;
  }
  const blocks = turns.map((turn) => {
    const red = redacted(turn);
    const action = turn.report ? turn.report.final_action : "aborted";
    const reply = turn.error
      ? `<div class="bubble error">The turn failed: ${esc(turn.error.type)}: ${esc(turn.error.message)}</div>`
      : `<div class="bubble">${esc(turn.reply)}</div>`;
    return `<article class="turn${inspected === turn.n ? " inspected" : ""}" aria-label="Turn ${turn.n}">
      <div class="turn-head">
        <button class="lens" type="button" data-act="inspect" data-n="${turn.n}" aria-label="Inspect turn ${turn.n}" title="Open the report of turn ${turn.n}">${LENS}<span>Turn ${turn.n}</span></button>
        <div class="msg user${red ? " redacted" : ""}"><div class="bubble">${esc(turn.message)}</div>${red ? `<div class="kept">Kept out of the history as <code>${esc(placeholder)}</code></div>` : ""}</div>
      </div>
      <div class="msg bot"><div class="avatar">${symbol}</div><div class="col">${reply}
        <div class="msg-meta">${trail(turn)}<span class="pill fa-${esc(action)}">${esc(action)}</span><span class="mono">${fmtMs(turn.report ? turn.report.duration_ms : 0)}</span>
        <button class="tbtn" type="button" data-act="rerun" data-n="${turn.n}"${busy ? " disabled" : ""}>Re-run</button></div></div></div>
    </article>`;
  });
  if (pending) {
    blocks.push(`<article class="turn" aria-label="Turn ${pending.n}"><div class="turn-head"><span></span>
      <div class="msg user"><div class="bubble">${esc(pending.message)}</div></div></div>
      <div class="msg bot"><div class="avatar">${symbol}</div><div class="bubble typing" aria-label="Waiting for the reply"><i></i><i></i><i></i></div></div></article>`);
  }
  log.innerHTML = blocks.join("");
}

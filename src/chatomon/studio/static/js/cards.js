import { $, esc, fmtMs, shortFp, plural } from "./util.js";
import { nucleusFailed, results } from "./turns.js";

const dlg = $("#dlg");
let onCloseHook = () => {};

export function openDialog({ title, sub = "", extra = "", body, actions, size = "s", onClose = () => {} }) {
  onCloseHook = onClose;
  $("#dlg-title").innerHTML = title;
  $("#dlg-sub").innerHTML = sub;
  $("#dlg-sub").hidden = !sub;
  $("#dlg-extra").innerHTML = extra;
  $("#dlg-body").innerHTML = body;
  $("#dlg-actions").innerHTML = actions;
  $("#dlg-extra").onclick = null;
  $("#dlg-actions").onclick = null;
  $("#dlg-body").onclick = null;
  $("#dlg-body").onchange = null;
  $("#dlg-body").oninput = null;
  dlg.dataset.size = size;
  $("#dlg-body").scrollTop = 0;
  if (!dlg.open) dlg.showModal();
}
export function closeDialog() { if (dlg.open) dlg.close(); }
dlg.addEventListener("close", () => onCloseHook());
dlg.addEventListener("click", (e) => {
  if (e.target === dlg || e.target.closest('[data-dlg="close"]')) closeDialog();
});

export const kv = (rows) => `<dl class="kv">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;

export function showAtom(state) {
  const a = state.atom;
  openDialog({
    title: esc(a.name),
    sub: `<code>${esc(a.source)}</code>`,
    body: kv([
      ["connector", `${a.connector} · ${a.model}`],
      ["instructions", a.instructions],
      ["knowledge", a.knowledge ?? "—"],
      ["max_retries", String(a.max_retries)],
      ["refusal", a.refusal],
      ["blocked_placeholder", a.blocked_placeholder],
      ["capture_content", `"${a.capture_content}" in your code · "full" in the studio`],
    ]) + `<p class="hint">The atom comes from your code. Change it there and restart the studio.</p>`,
    actions: `<span class="spacer"></span><button class="btn primary" type="button" data-dlg="close">Done</button>`,
  });
}

export function showNotice(local) {
  const where = local
    ? "The studio listens only on this machine."
    : "The studio listens on a network address: anyone who reaches it with the link can read this conversation and use the atom's models.";
  openDialog({
    title: "Content visible",
    body: `<p class="note">The studio runs a copy of your atom with <code>capture_content="full"</code>, so it can show message text, guidance and judge explanations. Your atom keeps its own setting.</p>
      <p class="note">${where}</p>
      <p class="note">The copy has its own turn callback and tracer: your <code>on_turn</code> and your tracer provider are not called.</p>`,
    actions: `<span class="spacer"></span><button class="btn primary" type="button" data-dlg="close">Done</button>`,
  });
}

function resRow(x) {
  if (x.action === "skip") {
    return `<div class="res skip"><div class="res-sum"><i class="res-dash skip"></i><span class="name">${esc(x.name)}</span><span class="pill skip">not run</span><span class="dur"></span></div></div>`;
  }
  const rows = [
    ["kind", x.kind],
    ["reason", x.reason ?? "—"],
    x.guidance && ["guidance", x.guidance],
    x.reply && ["reply", x.reply],
    x.annotations && ["annotations", x.annotations],
    x.error_type && ["error", `${x.error_type}${x.error_message ? `: ${x.error_message}` : ""} (on_error=${x.error_policy})`],
    ["fingerprint", shortFp(x.fingerprint)],
  ].filter(Boolean);
  const mono = new Set(["kind", "reason", "annotations", "fingerprint"]);
  return `<details class="res"${x.action !== "allow" ? " open" : ""}><summary class="res-sum"><i class="res-dash ${x.action === "block" ? "block" : ""}"></i><span class="name">${esc(x.name)}</span><span class="pill act-${esc(x.action)}">${esc(x.action)}</span><span class="dur">${fmtMs(x.duration_ms)}</span></summary>
    <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd${mono.has(k) ? ' class="mono"' : ""}${k === "fingerprint" ? ` title="${esc(x.fingerprint)}"` : ""}>${esc(v)}</dd>`).join("")}</dl></details>`;
}

export function showReport(turn, { total, onNav, onRerun, onClose, busy }) {
  const r = turn.report;
  const action = r ? r.final_action : "aborted";
  const attempts = r ? r.output : [];
  const input = `<section class="rep-sec"><h3 class="label">Input chain</h3>${turn.chain.input.length ? `<div class="res-list">${results(turn, "input").map(resRow).join("")}</div>` : `<p class="muted">No input orbitals were on.</p>`}</section>`;
  const output = attempts.length
    ? `<section class="rep-sec"><h3 class="label">Nucleus and output chain</h3>${attempts.map((_, i) => {
        const last = i === attempts.length - 1;
        const status = last ? (action === "answered" || action === "guided_refusal" ? "sent" : "replaced") : "rejected";
        const rows = results(turn, "output", i);
        return `<div class="attempt"><div class="attempt-head"><strong>Attempt ${i + 1}</strong><span class="pill ${status === "sent" ? "act-allow" : "act-block"}">${status}</span></div>${rows.length ? `<div class="res-list">${rows.map(resRow).join("")}</div>` : `<p class="muted">No output orbitals were on.</p>`}</div>`;
      }).join("")}</section>`
    : `<section class="rep-sec"><h3 class="label">Nucleus</h3><p class="muted">${nucleusFailed(turn)
        ? "The nucleus was called and failed before a reply could be checked."
        : `Not called: the input chain ${action === "orbital_response" ? "answered" : "stopped"} the turn first.`}</p></section>`;
  const sent = turn.error
    ? `<section class="rep-sec"><h3 class="label">Error</h3><div class="final error">${esc(turn.error.type)}: ${esc(turn.error.message)}</div></section>`
    : `<section class="rep-sec"><h3 class="label">Sent to the user</h3><div class="final">${esc(turn.reply)}</div></section>`;
  openDialog({
    size: "l",
    onClose,
    title: `Turn ${turn.n}<span class="pill fa-${esc(action)}">${esc(action)}</span>`,
    sub: `<span class="mono">${fmtMs(r ? r.duration_ms : 0)} · ${plural(turn.chain.input.length, "input orbital")} · ${plural(turn.chain.output.length, "output orbital")}</span>`,
    extra: `<span class="nav-btns"><button class="icon-btn" type="button" data-nav="-1" aria-label="Previous turn"${turn.n === 1 ? " disabled" : ""}>‹</button><button class="icon-btn" type="button" data-nav="1" aria-label="Next turn"${turn.n >= total ? " disabled" : ""}>›</button></span>`,
    body: `<p class="rep-msg">${esc(turn.message)}</p>${input}${output}${sent}<p class="foot">Text, guidance and judge explanations are visible because the studio runs with <code>capture_content="full"</code>. Your atom keeps its own setting.</p>`,
    actions: `<button class="btn" type="button" data-act="rerun"${busy ? " disabled" : ""}>Re-run with current orbitals</button><span class="spacer"></span><button class="btn" type="button" data-dlg="close">Close</button>`,
  });
  $("#dlg-extra").onclick = (e) => { const b = e.target.closest("[data-nav]"); if (b && !b.disabled) onNav(turn.n + Number(b.dataset.nav)); };
  $("#dlg-actions").onclick = (e) => { if (e.target.closest('[data-act="rerun"]')) { closeDialog(); onRerun(turn.message); } };
}

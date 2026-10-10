import { $, esc, fmtMs, rich, shortFp, plural } from "./util.js";
import { reloadAtom } from "./api.js";
import { nucleusFailed, results } from "./turns.js";

const dlg = $("#dlg");
let onCloseHook = () => {};

export function openDialog({ title, sub = "", extra = "", lead = "", body, actions, size = "s", onClose = () => {} }) {
  onCloseHook = onClose;
  // What a card shows beside its title, such as a drawing: its place is made when one asks for it.
  let leadBox = $("#dlg-lead");
  if (lead && !leadBox) {
    leadBox = document.createElement("div");
    leadBox.id = "dlg-lead";
    leadBox.className = "dlg-lead";
    $(".dlg-head", dlg).prepend(leadBox);
  }
  if (leadBox) {
    leadBox.innerHTML = lead;
    leadBox.hidden = !lead;
  }
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
  $("#dlg-body").onkeydown = null;
  dlg.dataset.size = size;
  $("#dlg-body").scrollTop = 0;
  if (!dlg.open) dlg.showModal();
  // A card drawn over another loses the control that had the focus: the title takes it, so the
  // keyboard and screen readers stay inside the card.
  else if (!dlg.contains(document.activeElement)) $("#dlg-title").focus();
  // A modal card blurs and blocks everything outside it: the toast goes inside, to stay
  // readable and usable (Undo, the reason of a refusal).
  dlg.append($("#toast"));
}
export function closeDialog() { if (dlg.open) dlg.close(); }
dlg.addEventListener("close", () => {
  document.body.append($("#toast"));
  onCloseHook();
});
// A click on the backdrop closes the card, but not the end of a drag that started inside it,
// such as selecting the text of a field.
let pressedOnBackdrop = false;
dlg.addEventListener("pointerdown", (e) => { pressedOnBackdrop = e.target === dlg; });
dlg.addEventListener("click", (e) => {
  if ((e.target === dlg && pressedOnBackdrop) || e.target.closest('[data-dlg="close"]')) closeDialog();
});

export const kv = (rows) => `<dl class="kv">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>`;

const RELOAD = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg>`;

/** What a reload did, in the words of the server: what was dropped and why a conversation started again. */
function reloaded(done) {
  const dropped = done.dropped.length
    ? `<p>No longer applied:</p><ul>${done.dropped.map((sentence) => `<li>${rich(sentence)}</li>`).join("")}</ul>`
    : "";
  const restarted = done.restarted ? `<p>${rich(done.restarted)}</p>` : "";
  const more = dropped + restarted;
  // With nothing to report the line is plain; otherwise it has the look of a note that warns.
  return `<div class="note${more ? " warn" : ""}" id="atom-outcome" role="status">${more ? "<p>" : ""}Reloaded from <code>${esc(done.state.atom.source)}</code>.${more ? "</p>" : ""}${more}</div>`;
}

/** Why a reload was not done. Only a refusal knows that the atom from before is still running. */
function failed(error) {
  const sentence = error.message.replace(/\.$/, "");
  const rest = error.status === 409 ? " The atom from before is still running; the terminal has the details." : "";
  return `<div class="note err" id="atom-outcome" role="alert">Reload failed: ${rich(sentence)}.${rest}</div>`;
}

// What a reload did while its card was not on screen: shown once, the next time the card opens.
let unseen = "";

/** A turn started or ended: the button of the atom card follows, and says why it cannot be used. */
export function syncReload(busy) {
  const button = $('[data-dlg="reload"]', dlg);
  if (!button || button.dataset.reloading) return;
  button.disabled = busy;
  button.title = busy ? "Wait for the turn to finish" : "";
}

/** The atom card. `outcome` is the box that says what a reload asked from this card did. */
export function showAtom(ctx, outcome = "") {
  const state = ctx.state();
  const a = state.atom;
  const can = state.studio.reload;
  const result = outcome || unseen;
  unseen = "";
  openDialog({
    title: esc(a.name),
    sub: `<code>${esc(a.source)}</code>`,
    body: result + kv([
      ["connector", `${a.connector} · ${a.model}`],
      ["instructions", a.instructions],
      ["knowledge", a.knowledge ?? "—"],
      ["max_retries", String(a.max_retries)],
      ["refusal", a.refusal],
      ["blocked_placeholder", a.blocked_placeholder],
      ["blocked_guidance", a.blocked_guidance ?? "—"],
      ["capture_content", `"${a.capture_content}" in your code · "full" in the studio`],
    ]) + `<p class="hint">The atom comes from your code. Change it there and ${can ? "press Reload: the studio keeps your orbital changes and the conversation." : "restart the studio."}</p>`,
    actions: `${can ? `<button class="btn" type="button" data-dlg="reload">${RELOAD}<span>Reload from code</span></button>` : ""}<span class="spacer"></span><button class="btn primary" type="button" data-dlg="close">Done</button>`,
  });
  const button = $('[data-dlg="reload"]', dlg);
  syncReload(ctx.busy());
  // The card was drawn again with the result: the focus goes back to the button that was used.
  if (outcome && button) button.focus();
  // The card can be closed, or another one opened, while the code loads.
  const shown = () => dlg.open && button.isConnected;
  const ready = () => {
    delete button.dataset.reloading;
    $("span", button).textContent = "Reload from code";
    syncReload(ctx.busy());
    if (shown() && !button.disabled) button.focus();
  };
  $("#dlg-actions").onclick = async (e) => {
    if (!button || !e.target.closest('[data-dlg="reload"]') || button.disabled) return;
    button.disabled = true;
    button.dataset.reloading = "1";
    $("span", button).textContent = "Reloading…";
    // What the box said was about the reload before this one.
    $("#atom-outcome", dlg)?.remove();
    let done;
    try {
      done = await reloadAtom();
    } catch (error) {
      // No answer at all: nothing is known of the atom, and the page has its own words for it.
      if (error.status === undefined) { ready(); ctx.lost(); return; }
      // A turn or another reload runs: nothing failed, the card stays as it is.
      if (error.type === "Busy") { ready(); ctx.toast(error.message); return; }
      if (shown()) { showAtom(ctx, failed(error)); return; }
      unseen = failed(error);
      ctx.toast(`Reload failed: ${error.message}`);
      return;
    }
    ctx.refresh(done.state);
    if (shown()) { showAtom(ctx, reloaded(done)); return; }
    // The result does not take the screen from what is open now: it is said, and kept for the card.
    const more = done.dropped.length || done.restarted;
    unseen = more ? reloaded(done) : "";
    const dropped = done.dropped.length ? ` ${plural(done.dropped.length, "change")} no longer applied: the atom card says which.` : "";
    ctx.toast(`Reloaded from code.${dropped}${done.restarted ? ` ${done.restarted}` : ""}`);
  };
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

// What the engine decided about a reply of the nucleus, as the report says it; an attempt
// with no decision is one the turn was aborted in.
const DECIDED = { accept: "accepted", replace: "replaced", regenerate: "rejected" };

export function showReport(turn, { total, onNav, onRerun, onSave, onClose, busy }) {
  const r = turn.report;
  const action = r ? r.final_action : "aborted";
  const attempts = r ? r.output : [];
  const input = `<section class="rep-sec"><h3 class="label">Input chain</h3>${turn.chain.input.length ? `<div class="res-list">${results(turn, "input").map(resRow).join("")}</div>` : `<p class="muted">No input orbitals were on.</p>`}</section>`;
  const output = attempts.length
    ? `<section class="rep-sec"><h3 class="label">Nucleus and output chain</h3>${attempts.map((attempt, i) => {
        const status = DECIDED[attempt.decision] || "interrupted";
        const rows = results(turn, "output", i);
        // What the nucleus wrote, whatever was decided about it: the text that replaced a reply
        // is under "Sent to the user".
        // A reply of blank space would be an empty box, which reads as a fault of the page.
        const text = attempt.nucleus_reply || "";
        const wrote = text.trim()
          ? `<p class="draft"><span class="sr">The nucleus wrote: </span>${esc(text)}</p>`
          : `<p class="muted">The nucleus wrote ${text ? "only blank space" : "nothing"}.</p>`;
        return `<div class="attempt"><div class="attempt-head"><strong>Attempt ${i + 1}</strong><span class="pill ${attempt.decision === "accept" ? "act-allow" : "act-block"}">${status}</span></div>${wrote}${rows.length ? `<div class="res-list">${rows.map(resRow).join("")}</div>` : `<p class="muted">No output orbitals were on.</p>`}</div>`;
      }).join("")}</section>`
    : `<section class="rep-sec"><h3 class="label">Nucleus</h3><p class="muted">${nucleusFailed(turn)
        ? "The nucleus was called and failed before a reply could be checked."
        : `Not called: the input chain ${action === "orbital_response" ? "answered" : "stopped"} the turn first.`}</p></section>`;
  const sent = turn.error
    ? `<section class="rep-sec"><h3 class="label">Error</h3><div class="final error">${esc(turn.error.type)}: ${esc(turn.error.message)}</div></section>`
    : `<section class="rep-sec"><h3 class="label">Sent to the user</h3><div class="final">${esc(turn.reply)}</div></section>`;
  const changes = turn.changes.length ? `${plural(turn.changes.length, "change")} vs code` : "same as code";
  // A turn that cannot be saved says why next to its disabled button: a tooltip is not reachable by touch.
  const save = `<button class="btn primary" type="button" data-act="save"${turn.cannot_save ? " disabled" : ""}>Save as test</button>`;
  const why = turn.cannot_save ? `<span class="hint">${esc(turn.cannot_save)}</span>` : `<span class="spacer"></span>`;
  openDialog({
    size: "l",
    onClose,
    title: `Turn ${turn.n}<span class="pill fa-${esc(action)}">${esc(action)}</span>`,
    sub: `<span class="mono">${fmtMs(r ? r.duration_ms : 0)} · ${plural(turn.chain.input.length, "input orbital")} · ${plural(turn.chain.output.length, "output orbital")} · ${changes}</span>`,
    extra: `<span class="nav-btns"><button class="icon-btn" type="button" data-nav="-1" aria-label="Previous turn"${turn.n === 1 ? " disabled" : ""}>‹</button><button class="icon-btn" type="button" data-nav="1" aria-label="Next turn"${turn.n >= total ? " disabled" : ""}>›</button></span>`,
    body: `<p class="rep-msg">${esc(turn.message)}</p>${input}${output}${sent}<p class="foot">Text, guidance and judge explanations are visible because the studio runs with <code>capture_content="full"</code>. Your atom keeps its own setting.</p>`,
    actions: `${save}<button class="btn" type="button" data-act="rerun"${busy ? " disabled" : ""}>Re-run with current orbitals</button>${why}<button class="btn" type="button" data-dlg="close">Close</button>`,
  });
  $("#dlg-extra").onclick = (e) => { const b = e.target.closest("[data-nav]"); if (b && !b.disabled) onNav(turn.n + Number(b.dataset.nav)); };
  $("#dlg-actions").onclick = (e) => {
    if (e.target.closest('[data-act="rerun"]')) { closeDialog(); onRerun(turn.message); }
    if (e.target.closest('[data-act="save"]')) onSave(turn.n);
  };
}

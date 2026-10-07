import { $, $$, esc, fmtMs, rich, shortFp } from "./util.js";
import { addOrbital, patchOrbital, probeOrbital, resetOrbitals } from "./api.js";
import { HELP, TRASH, block, breaksLine, handle, hydrate, keep, mark, read as readBlock } from "./blocks.js";
import { closeDialog, kv, openDialog } from "./cards.js";

const dlg = $("#dlg");
let ctx = null;
let view = { side: null, name: null };
// The block being applied, so that a test started by the same click waits for it.
let settling = Promise.resolve(true);
let pressedBlock = null;
// The card whose test is on its way: its outcome goes to the row that is there when it arrives.
let testing = null;
const cardKey = () => ($("#card-switch", dlg) ? `${view.side}/${view.name}` : "form");

const find = (state, side, name) => state.orbitals[side].find((o) => o.name === name);
const fieldId = (key) => `f-${key}`;
const errorOf = (el) => $(`#${CSS.escape(el.id)}-err`, dlg);
const others = (state, name) => [...new Set([...state.orbitals.input, ...state.orbitals.output].map((o) => o.name))].filter((n) => n !== name);

/** One control per setting type; the page never decides what a setting is. */
function control(s, value, names) {
  const id = fieldId(s.key);
  const attrs = `id="${id}" data-key="${esc(s.key)}" data-type="${s.type}" aria-describedby="${id}-err"`;
  // A choice or a switch that may be none gets a "None" entry; a switch then becomes a select.
  const empty = (selected) => `<option value=""${selected ? " selected" : ""}>None</option>`;
  const option = (o, selected) => `<option${selected ? " selected" : ""}>${esc(o)}</option>`;
  if (s.type === "choice") return `<select ${attrs}>${s.optional ? empty(value == null) : ""}${s.options.map((o) => option(o, o === value)).join("")}</select>`;
  if (s.type === "bool" && s.optional) return `<select ${attrs}>${empty(value == null)}${option("true", value === true)}${option("false", value === false)}</select>`;
  if (s.type === "bool") return `<label class="switch"><input type="checkbox" role="switch" ${attrs}${value ? " checked" : ""}><span></span></label>`;
  if (s.type === "names") {
    const boxes = names.map((n) => `<label><input type="checkbox" value="${esc(n)}"${(value || []).includes(n) ? " checked" : ""}>${esc(n)}</label>`).join("");
    return `<fieldset ${attrs}><legend>${esc(s.key)}</legend>${boxes || `<span class="muted">No other orbitals.</span>`}</fieldset>`;
  }
  const none = s.optional ? ' placeholder="None"' : "";
  if (s.type === "text") return `<textarea ${attrs} rows="1"${none}>${esc(value ?? "")}</textarea>`;
  if (s.type === "env") return `<input type="text" class="mono" ${attrs} value="${esc(value ?? "")}" autocapitalize="none" autocomplete="off" spellcheck="false"${none}>`;
  // A text field, not a number field: whatever is typed reaches the server, which says if it is a number.
  return `<input type="text" inputmode="${s.type === "int" ? "numeric" : "decimal"}" autocomplete="off" ${attrs} value="${esc(value ?? "")}"${none}>`;
}

function field(s, value, names, changed = false) {
  const wide = ["text", "names", "env"].includes(s.type);
  const label = s.type === "names" ? "" : `<label for="${fieldId(s.key)}">${esc(s.key)}</label>`;
  const help = s.type === "env" ? `<span class="help">${HELP.env}</span>` : "";
  return `<div class="field${wide ? " wide" : ""}${changed ? " changed" : ""}">${label}${control(s, value, names)}${help}<p class="field-err" id="${fieldId(s.key)}-err" role="alert" hidden></p></div>`;
}

const nameField = (value) => `<div class="field wide"><label for="f-name">name</label><input type="text" id="f-name" data-key="name" data-type="name" value="${esc(value)}" autocapitalize="none" autocomplete="off" spellcheck="false" aria-describedby="f-name-err"><p class="field-err" id="f-name-err" role="alert" hidden></p></div>`;

/** The JSON value of a control; an empty field means none, and the server says if that is allowed. */
function read(el) {
  const type = el.dataset.type;
  if (el.tagName === "SELECT" && el.value === "") return null;
  if (type === "bool") return el.tagName === "SELECT" ? el.value === "true" : el.checked;
  if (type === "names") return $$("input:checked", el).map((box) => box.value);
  if (type === "choice" || type === "name") return el.value;
  if (el.value.trim() === "") return null;
  if (type === "text" || type === "env") return el.value;
  const number = Number(el.value.trim());
  return Number.isFinite(number) ? number : el.value;
}

function fail(el, message) {
  const err = errorOf(el);
  err.innerHTML = rich(message);
  err.hidden = false;
  el.setAttribute("aria-invalid", "true");
}
function clear(el) {
  errorOf(el).hidden = true;
  el.removeAttribute("aria-invalid");
}
const grow = (el) => { el.style.height = "auto"; el.style.height = `${el.scrollHeight + 2}px`; };

/** A setting as its field, or as a block when its value has fields of its own. */
const drawn = (s, value, names, changed = false) => (s.type === "block" ? block(s, value, `b-${s.key}`, changed) : field(s, value, names, changed));

const PROBE = `<div class="card-row"><span class="label">Connection</span><span class="probe" id="probe-out" role="status">Not tested yet</span><button class="btn sm" type="button" data-dlg="probe">Test connection</button></div>
  <p class="hint">Sends one sample message, as a turn would. Nothing is added or saved.</p>`;

/** Show, in the row that is in the card now, that a test is on its way. */
function waiting() {
  const button = $('[data-dlg="probe"]', dlg);
  if (!button) return;
  button.disabled = true;
  button.textContent = "Testing…";
  $("#probe-out", dlg).className = "probe";
  $("#probe-out", dlg).textContent = "Sending one sample message…";
}

/** Run a test and show how it went, in the words of the server. */
async function probe(request) {
  const key = cardKey();
  const form = $("#probe-out", dlg);
  testing = key;
  waiting();
  // A card can be drawn again while the test runs, as when its switch is pressed: the outcome
  // goes to the row of the same orbital that is there now. A form left meanwhile gets nothing.
  const row = () => (key === "form" ? (form.isConnected ? form : null) : dlg.open && cardKey() === key ? $("#probe-out", dlg) : null);
  let out;
  try {
    const result = await request();
    testing = null;
    out = row();
    if (!out) return;
    if (result.outcome === "answered") {
      out.innerHTML = `<span class="pill act-${esc(result.action)}">${esc(result.action)}</span>${result.reason ? `<span>${esc(result.reason)}</span>` : ""}<span class="mono">${fmtMs(result.duration_ms)}</span>`;
    } else {
      out.className = "probe err";
      out.textContent = result.message;
    }
  } catch (error) {
    testing = null;
    out = row();
    if (!out) return;
    // A refusal before the check, such as a variable that is not set, is shown where the outcome is.
    if (![404, 409, 422].includes(error.status)) { closeDialog(); await ctx.refused(error); return; }
    out.className = "probe err";
    out.innerHTML = rich(error.message);
  } finally {
    const button = row() && $('[data-dlg="probe"]', dlg);
    if (button) { button.disabled = false; button.textContent = "Test connection"; }
  }
}

// In an orbital's card a block is applied whole when the focus leaves it. A press inside it
// that takes the focus nowhere, as on its padding, is not leaving it. These two are listeners
// and not properties of the card's body: `focusout` has no property to assign a handler to.
dlg.addEventListener("pointerdown", (e) => { pressedBlock = e.target.closest("fieldset.block"); });
dlg.addEventListener("focusout", (e) => {
  const fs = e.target.closest("fieldset.block");
  if (!fs || !fs.classList.contains("dirty") || !$("#card-switch", dlg)) return;
  if (e.relatedTarget ? fs.contains(e.relatedTarget) : pressedBlock === fs) return;
  settling = commitBlock(fs);
});

/** A block starts to differ from what the server has: say so until it is applied. */
function touch(fs) {
  fs.classList.add("dirty");
  fs._unsent = true;
  $(":scope > .block-body > .block-state", fs).hidden = false;
}

/**
 * Apply the blocks whose edits were never sent. Leaving a block applies it, but a press that
 * takes the focus nowhere leaves it waiting: what needs the card as the server has it asks here.
 * Resolves to whether no block is left edited, which a refused one is.
 */
async function applied() {
  await settling;
  for (const fs of $$("fieldset.block.dirty", dlg)) {
    if (!fs._unsent) continue;
    settling = commitBlock(fs);
    await settling;
  }
  return !$("fieldset.block.dirty", dlg);
}

function draw(state) {
  const o = find(state, view.side, view.name);
  if (!o) { closeDialog(); return; }
  const list = state.orbitals[o.side];
  const i = list.indexOf(o);
  const added = o.origin === "studio";
  const source = `<code>${esc(state.atom.source)}</code>`;
  const names = others(state, o.name);
  const fixed = o.fixed.length ? `<dl class="kv">${o.fixed.map((f) => `<dt>${esc(f.key)}</dt><dd>${esc(f.name ? `${f.name} (${f.type})` : f.type)} <span class="from">· from code</span></dd>`).join("")}</dl>` : "";
  const config = Object.entries(o.config).filter(([k]) => k !== "kind" && k !== "name").map(([k, v]) => [k, typeof v === "string" ? v : JSON.stringify(v)]);
  const body = o.editable
    ? `${fixed}<div class="fields">${added ? nameField(o.name) : ""}${o.settings.map((s) => drawn(s, s.value, names, s.changed)).join("")}</div>${o.fixed.length ? `<p class="hint">Settings marked “from code” are objects in your code. The studio changes only the others.</p>` : ""}`
    : `${kv(config)}<p class="hint">${esc(o.note)} Here you can turn it off or move it in the chain.</p>`;
  const position = `<div class="card-row"><span class="label">Position</span><span class="pos" id="card-pos">${i + 1} of ${list.length} in the ${o.side} chain</span><button class="btn sm" type="button" data-dlg="up"${i === 0 ? " disabled" : ""}>Move up</button><button class="btn sm" type="button" data-dlg="down"${i === list.length - 1 ? " disabled" : ""}>Move down</button></div>`;
  const left = added
    ? `<button class="btn danger" type="button" data-dlg="delete">${TRASH}Delete orbital</button>`
    : `<span class="hint">Defined in your code: turn it off here, delete it in ${source}.</span>`;
  openDialog({
    title: `<span class="dlg-glyph${o.enabled ? "" : " off"}"></span><span class="mono" id="card-title">${esc(o.name)}</span>`,
    sub: `${esc(o.kind)} · ${o.side} orbital · ${added ? "added in the studio" : `from ${source}`}`,
    extra: `<span class="sw-label">${o.enabled ? "On" : "Off"}</span><label class="switch big"><input type="checkbox" role="switch" id="card-switch"${o.enabled ? " checked" : ""} aria-label="${esc(o.name)}"><span></span></label>`,
    body: `${body}${o.calls_model ? PROBE : ""}${position}<div class="fp" id="card-fp" title="${esc(o.fingerprint)}">fingerprint ${esc(shortFp(o.fingerprint))} · changes with every setting and is recorded in each turn report</div>`,
    actions: `${left}<span class="spacer"></span>${added ? "" : `<button class="btn" type="button" data-dlg="reset-one"${o.modified ? "" : " disabled"}>Reset to code</button>`}<button class="btn primary" type="button" data-dlg="close">Done</button>`,
    onClose: () => {
      // Closing the card leaves its blocks: one never sent is applied, and a refusal goes to the toast.
      for (const fs of $$("fieldset.block.dirty", dlg)) if (fs._unsent) commitBlock(fs);
      ctx.closed(view.side, view.name);
    },
  });
  $$("textarea", dlg).forEach(grow);
  hydrate(dlg);
  settling = Promise.resolve(true);
  if (testing === cardKey()) waiting();
  $("#card-switch").onchange = (e) => act(() => patchOrbital(view.side, view.name, { enabled: e.target.checked }), "#card-switch");
  $("#dlg-body").oninput = (e) => {
    if (e.target.tagName === "TEXTAREA") grow(e.target);
    const fs = e.target.closest("fieldset.block");
    if (fs) touch(fs);
  };
  $("#dlg-body").onchange = (e) => {
    const changed = handle(e);
    if (changed) { touch(changed); return; }
    const el = e.target.closest("[data-key]");
    if (!el) return;
    // Enter already sent this value: leaving the field afterwards is not a second change.
    if (el.dataset.entered === el.value) { delete el.dataset.entered; return; }
    commit(el);
  };
  // In a text area Enter applies the setting, as in the other fields; Shift+Enter breaks the line.
  // In a block Enter applies the block, except where a list is written one entry per line.
  $("#dlg-body").onkeydown = (e) => {
    if (e.key !== "Enter" || e.shiftKey || e.isComposing) return;
    const fs = e.target.closest("fieldset.block");
    if (fs) {
      if (!["TEXTAREA", "INPUT"].includes(e.target.tagName) || breaksLine(e.target)) return;
      e.preventDefault();
      if (fs.classList.contains("dirty")) settling = commitBlock(fs);
      return;
    }
    if (e.target.tagName !== "TEXTAREA") return;
    e.preventDefault();
    e.target.dataset.entered = e.target.value;
    commit(e.target);
  };
  $("#dlg-body").onclick = async (e) => {
    const changed = handle(e);
    if (changed) { touch(changed); return; }
    if (e.target.closest('[data-dlg="probe"]')) {
      // A block edited and not applied is applied first; a refused one leaves nothing to test.
      const { side, name } = view;
      if (await applied()) await probe(() => probeOrbital(side, { name }));
      return;
    }
    const b = e.target.closest('[data-dlg="up"], [data-dlg="down"]');
    if (!b || b.disabled) return;
    const up = b.dataset.dlg === "up";
    const j = up ? i - 1 : i + 1;
    // Keep the focus on the same button, or on the other one when this one ends up disabled.
    const stays = up ? j > 0 : j < list.length - 1;
    act(() => patchOrbital(view.side, view.name, { position: j }), `[data-dlg="${stays === up ? "up" : "down"}"]`);
  };
  $("#dlg-actions").onclick = async (e) => {
    if (e.target.closest('[data-dlg="delete"]')) {
      // There is nothing to apply to an orbital that goes.
      for (const fs of $$("fieldset.block.dirty", dlg)) fs._unsent = false;
      closeDialog();
      ctx.remove(view.side, view.name);
    }
    if (e.target.closest('[data-dlg="reset-one"]')) {
      await act(() => resetOrbitals({ side: view.side, name: view.name }), '[data-dlg="close"]', { keeping: false });
      ctx.offerUndo();
    }
  };
}

/**
 * A change that redraws the whole card: the switch, a move, a reset. The blocks are settled
 * first, and what the orbital refused, a field or a block, comes back as it was typed, with
 * its refusal: only a reset, which is there to discard, leaves it behind.
 */
async function act(request, focus, { keeping = true } = {}) {
  if (keeping) await applied();
  const refused = keeping ? $$("fieldset.block.dirty", dlg).map((fs) => ({ key: fs.dataset.block, typed: readBlock(fs), refusal: fs._refusal })) : [];
  const fields = keeping ? $$('[data-key][aria-invalid="true"]', dlg).map((el) => ({ id: el.id, typed: el.dataset.type === "names" ? read(el) : el.value, refusal: errorOf(el).innerHTML })) : [];
  try {
    draw(await ctx.apply(request));
  } catch (error) {
    await ctx.refused(error);
    draw(ctx.state());
  }
  for (const { key, typed, refusal } of refused) {
    const old = $(`fieldset.block[data-block="${CSS.escape(key)}"]`, dlg);
    if (!old) continue;
    const { _setting: setting, _value: held } = old;
    old.outerHTML = block(setting, typed, old.dataset.pid, old.classList.contains("changed"));
    hydrate(dlg);
    const fs = $(`fieldset.block[data-block="${CSS.escape(key)}"]`, dlg);
    // What the server holds is still what it held: choosing its form again shows that.
    keep(fs, held);
    touch(fs);
    fs._unsent = false;
    mark(fs, refusal);
  }
  for (const { id, typed, refusal } of fields) {
    const el = $(`#${CSS.escape(id)}`, dlg);
    if (!el) continue;
    if (el.dataset.type === "names") $$("input", el).forEach((box) => { box.checked = typed.includes(box.value); });
    else el.value = typed;
    // The message is the one already shown: it was written as the page writes a refusal.
    errorOf(el).innerHTML = refusal;
    errorOf(el).hidden = false;
    el.setAttribute("aria-invalid", "true");
  }
  if (focus) $(focus, dlg)?.focus();
}

/** Send one field. The card is not redrawn, so what is being typed elsewhere stays. */
async function commit(el) {
  const key = el.dataset.key;
  const value = read(el);
  const { side, name } = view;
  clear(el);
  let state;
  try {
    state = await ctx.apply(() => patchOrbital(side, name, { settings: { [key]: value } }));
  } catch (error) {
    // 422: the value; 409: the chain refuses it. Both belong under the field. When the card
    // was closed in the meantime (Done right after typing), the refusal is said in the toast.
    if (error.status === 422 || error.status === 409) {
      if (dlg.open && el.isConnected) fail(el, error.message);
      else ctx.toast(`${name}: ${error.message}`);
      return;
    }
    closeDialog();
    await ctx.refused(error);
    return;
  }
  if (!dlg.open || !el.isConnected) return;
  if (key === "name") view.name = value;
  settled(state);
}

/** After an accepted change the card is not drawn again: only what derives from it changes. */
function settled(state) {
  const o = find(state, view.side, view.name);
  if (!o) { closeDialog(); return; }
  $("#card-title").textContent = o.name;
  $("#card-switch").setAttribute("aria-label", o.name);
  $("#card-fp").title = o.fingerprint;
  $("#card-fp").textContent = `fingerprint ${shortFp(o.fingerprint)} · changes with every setting and is recorded in each turn report`;
  const reset = $('[data-dlg="reset-one"]', dlg);
  if (reset) reset.disabled = !o.modified;
  for (const s of o.settings) {
    const marked = s.type === "block" ? $(`fieldset.block[data-block="${CSS.escape(s.key)}"]`, dlg) : $(`#${CSS.escape(fieldId(s.key))}`, dlg)?.closest(".field");
    marked?.classList.toggle("changed", s.changed);
    if (marked && s.type === "block") keep(marked, s.value);
  }
}

/** Send a block whole. Resolves to whether it was accepted. */
async function commitBlock(fs) {
  const key = fs.dataset.block;
  const { side, name } = view;
  mark(fs, null);
  fs._unsent = false;
  let state;
  try {
    state = await ctx.apply(() => patchOrbital(side, name, { settings: { [key]: readBlock(fs) } }));
  } catch (error) {
    // As for a field: a refusal stays under the block, or goes to the toast when the card closed.
    if (error.status === 422 || error.status === 409) {
      if (dlg.open && fs.isConnected) mark(fs, error.message);
      else ctx.toast(`${name}: ${error.message}`);
      return false;
    }
    closeDialog();
    await ctx.refused(error);
    return false;
  }
  if (!dlg.open || !fs.isConnected) return true;
  fs.classList.remove("dirty");
  $(":scope > .block-body > .block-state", fs).hidden = true;
  settled(state);
  return true;
}

/** Draw the open orbital card again after a change made outside it, such as Undo. */
export function redrawOrbital(state) {
  if (dlg.open && $("#card-switch", dlg)) draw(state);
}

export function showOrbital(side, name, context) {
  ctx = context;
  view = { side, name };
  draw(ctx.state());
}

export function showPicker(side, context) {
  ctx = context;
  const kinds = ctx.state().kinds.filter((kind) => kind.sides.includes(side));
  openDialog({
    title: `Add an ${side} orbital`,
    sub: side === "input" ? "It runs on the user's message, after the orbitals already in the chain." : "It runs on the nucleus reply, after the orbitals already in the chain.",
    body: `<div class="picks">${kinds.map((kind) => `<button class="pick" type="button" data-kind="${esc(kind.kind)}"><span class="orb-name">${esc(kind.kind)}</span><span class="orb-kind">${esc(kind.summary)}</span></button>`).join("")}</div>
      ${ctx.state().studio.extras.map((missing) => `<p class="hint">More orbitals with <code>pip install 'chatomon[${esc(missing.extra)}]'</code></p>`).join("")}
      <p class="hint">Orbitals that need your own objects, such as functions, are added in code.</p>`,
    actions: `<span class="spacer"></span><button class="btn" type="button" data-dlg="close">Cancel</button>`,
  });
  $("#dlg-body").onclick = (e) => {
    const b = e.target.closest("[data-kind]");
    if (b) showAddForm(side, kinds.find((kind) => kind.kind === b.dataset.kind));
  };
}

function showAddForm(side, kind) {
  const names = others(ctx.state(), null);
  openDialog({
    title: `Add <span class="mono">${esc(kind.kind)}</span>`,
    sub: `${side} orbital · it is created when you press Add`,
    body: `<form id="add-form" class="fields" novalidate>${nameField(kind.name)}${kind.settings.map((s) => drawn(s, s.default, names)).join("")}</form>
      <p class="field-err" id="add-err" role="alert" hidden></p>${kind.calls_model ? PROBE : ""}`,
    actions: `<button class="btn" type="button" data-dlg="back">Back</button><span class="spacer"></span><button class="btn" type="button" data-dlg="close">Cancel</button><button class="btn primary" type="submit" form="add-form">Add</button>`,
  });
  $$("textarea", dlg).forEach(grow);
  hydrate(dlg);
  /** What the form holds now: the plain settings and, whole, the blocks. */
  const gathered = () => {
    const settings = {};
    for (const el of $$("[data-key]", dlg)) { clear(el); settings[el.dataset.key] = read(el); }
    for (const fs of $$("fieldset.block", dlg)) { mark(fs, null); settings[fs.dataset.block] = readBlock(fs); }
    return settings;
  };
  $("#dlg-body").oninput = (e) => { if (e.target.tagName === "TEXTAREA") grow(e.target); };
  $("#dlg-body").onchange = (e) => { handle(e); };
  $("#dlg-body").onclick = (e) => {
    if (handle(e)) return;
    if (!e.target.closest('[data-dlg="probe"]')) return;
    // The form as it is, before anything is added; the name plays no part in the check.
    const { name: _, ...settings } = gathered();
    probe(() => probeOrbital(side, { kind: kind.kind, settings }));
  };
  $("#dlg-actions").onclick = (e) => { if (e.target.closest('[data-dlg="back"]')) showPicker(side, ctx); };
  // Enter adds the orbital from a text area too; Shift+Enter breaks the line, and so does
  // Enter where a list is written one entry per line.
  $("#dlg-body").onkeydown = (e) => {
    if (e.key !== "Enter" || e.shiftKey || e.isComposing || e.target.tagName !== "TEXTAREA" || breaksLine(e.target)) return;
    e.preventDefault();
    $("#add-form").requestSubmit();
  };
  const first = kind.settings.find((s) => s.required);
  ((first && first.type === "block" ? $(`fieldset.block[data-block="${CSS.escape(first.key)}"] :is(textarea, input, select)`, dlg) : $(`#${CSS.escape(fieldId((first || {}).key || "name"))}`, dlg)) || $("#f-name")).focus();
  $("#add-form").onsubmit = async (e) => {
    e.preventDefault();
    const settings = gathered();
    $("#add-err").hidden = true;
    try {
      await ctx.apply(() => addOrbital(side, kind.kind, settings));
    } catch (error) {
      const fs = error.setting ? $(`fieldset.block[data-block="${CSS.escape(error.setting)}"]`, dlg) : null;
      if (fs) { mark(fs, error.message); $(":is(textarea, input, select)", fs)?.focus(); return; }
      const el = error.setting ? $(`#${CSS.escape(fieldId(error.setting))}`, dlg) : null;
      if (el) { fail(el, error.message); el.focus(); return; }
      $("#add-err").innerHTML = rich(error.message);
      $("#add-err").hidden = false;
      return;
    }
    view = { side, name: settings.name };
    draw(ctx.state());
    ctx.toast(`${settings.name} added to the ${side} chain`);
  };
}

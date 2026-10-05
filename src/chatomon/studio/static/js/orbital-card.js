import { $, $$, esc, shortFp } from "./util.js";
import { addOrbital, patchOrbital, resetOrbitals } from "./api.js";
import { closeDialog, kv, openDialog } from "./cards.js";

const TRASH = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/></svg>`;
const dlg = $("#dlg");
let ctx = null;
let view = { side: null, name: null };

const find = (state, side, name) => state.orbitals[side].find((o) => o.name === name);
const fieldId = (key) => `f-${key}`;
const errorOf = (el) => $(`#${CSS.escape(el.id)}-err`, dlg);
const others = (state, name) => [...new Set([...state.orbitals.input, ...state.orbitals.output].map((o) => o.name))].filter((n) => n !== name);

/** One control per setting type; the page never decides what a setting is. */
function control(s, value, names) {
  const id = fieldId(s.key);
  const attrs = `id="${id}" data-key="${esc(s.key)}" data-type="${s.type}" aria-describedby="${id}-err"`;
  if (s.type === "choice") return `<select ${attrs}>${s.options.map((o) => `<option${o === value ? " selected" : ""}>${esc(o)}</option>`).join("")}</select>`;
  if (s.type === "bool") return `<label class="switch"><input type="checkbox" role="switch" ${attrs}${value ? " checked" : ""}><span></span></label>`;
  if (s.type === "names") {
    const boxes = names.map((n) => `<label><input type="checkbox" value="${esc(n)}"${(value || []).includes(n) ? " checked" : ""}>${esc(n)}</label>`).join("");
    return `<fieldset ${attrs}><legend>${esc(s.key)}</legend>${boxes || `<span class="muted">No other orbitals.</span>`}</fieldset>`;
  }
  const none = s.optional ? ' placeholder="None"' : "";
  if (s.type === "text") return `<textarea ${attrs} rows="1"${none}>${esc(value ?? "")}</textarea>`;
  return `<input type="number" ${attrs} step="${s.type === "int" ? "1" : "any"}" value="${esc(value ?? "")}"${none}>`;
}

function field(s, value, names, changed = false) {
  const wide = s.type === "text" || s.type === "names";
  const label = s.type === "names" ? "" : `<label for="${fieldId(s.key)}">${esc(s.key)}</label>`;
  return `<div class="field${wide ? " wide" : ""}${changed ? " changed" : ""}">${label}${control(s, value, names)}<p class="field-err" id="${fieldId(s.key)}-err" role="alert" hidden></p></div>`;
}

const nameField = (value) => `<div class="field wide"><label for="f-name">name</label><input type="text" id="f-name" data-key="name" data-type="name" value="${esc(value)}" autocapitalize="none" autocomplete="off" spellcheck="false" aria-describedby="f-name-err"><p class="field-err" id="f-name-err" role="alert" hidden></p></div>`;

/** The JSON value of a control; an empty field means none, and the server says if that is allowed. */
function read(el) {
  const type = el.dataset.type;
  if (type === "bool") return el.checked;
  if (type === "names") return $$("input:checked", el).map((box) => box.value);
  if (type === "choice" || type === "name") return el.value;
  if (el.value.trim() === "") return null;
  return type === "text" ? el.value : Number(el.value);
}

function fail(el, message) {
  const err = errorOf(el);
  err.textContent = message;
  err.hidden = false;
  el.setAttribute("aria-invalid", "true");
}
function clear(el) {
  errorOf(el).hidden = true;
  el.removeAttribute("aria-invalid");
}
const grow = (el) => { el.style.height = "auto"; el.style.height = `${el.scrollHeight + 2}px`; };

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
    ? `${fixed}<div class="fields">${added ? nameField(o.name) : ""}${o.settings.map((s) => field(s, s.value, names, s.changed)).join("")}</div>${o.fixed.length ? `<p class="hint">Settings marked “from code” are objects in your code. The studio changes only the others.</p>` : ""}`
    : `${kv(config)}<p class="hint">${esc(o.note)} Here you can turn it off or move it in the chain.</p>`;
  const position = `<div class="card-row"><span class="label">Position</span><span class="pos" id="card-pos">${i + 1} of ${list.length} in the ${o.side} chain</span><button class="btn sm" type="button" data-dlg="up"${i === 0 ? " disabled" : ""}>Move up</button><button class="btn sm" type="button" data-dlg="down"${i === list.length - 1 ? " disabled" : ""}>Move down</button></div>`;
  const left = added
    ? `<button class="btn danger" type="button" data-dlg="delete">${TRASH}Delete orbital</button>`
    : `<span class="hint">Defined in your code: turn it off here, delete it in ${source}.</span>`;
  openDialog({
    title: `<span class="dlg-glyph${o.enabled ? "" : " off"}"></span><span class="mono" id="card-title">${esc(o.name)}</span>`,
    sub: `${esc(o.kind)} · ${o.side} orbital · ${added ? "added in the studio" : `from ${source}`}`,
    extra: `<span class="sw-label">${o.enabled ? "On" : "Off"}</span><label class="switch big"><input type="checkbox" role="switch" id="card-switch"${o.enabled ? " checked" : ""} aria-label="${esc(o.name)}"><span></span></label>`,
    body: `${body}${position}<div class="fp" id="card-fp" title="${esc(o.fingerprint)}">fingerprint ${esc(shortFp(o.fingerprint))} · changes with every setting and is recorded in each turn report</div>`,
    actions: `${left}<span class="spacer"></span>${added ? "" : `<button class="btn" type="button" data-dlg="reset-one"${o.modified ? "" : " disabled"}>Reset to code</button>`}<button class="btn primary" type="button" data-dlg="close">Done</button>`,
  });
  $$("textarea", dlg).forEach(grow);
  $("#card-switch").onchange = (e) => act(() => patchOrbital(view.side, view.name, { enabled: e.target.checked }), "#card-switch");
  $("#dlg-body").oninput = (e) => { if (e.target.tagName === "TEXTAREA") grow(e.target); };
  $("#dlg-body").onchange = (e) => { const el = e.target.closest("[data-key]"); if (el) commit(el); };
  $("#dlg-body").onclick = (e) => {
    const b = e.target.closest('[data-dlg="up"], [data-dlg="down"]');
    if (!b || b.disabled) return;
    const up = b.dataset.dlg === "up";
    const j = up ? i - 1 : i + 1;
    // Keep the focus on the same button, or on the other one when this one ends up disabled.
    const stays = up ? j > 0 : j < list.length - 1;
    act(() => patchOrbital(view.side, view.name, { position: j }), `[data-dlg="${stays === up ? "up" : "down"}"]`);
  };
  $("#dlg-actions").onclick = async (e) => {
    if (e.target.closest('[data-dlg="delete"]')) { closeDialog(); ctx.remove(view.side, view.name); }
    if (e.target.closest('[data-dlg="reset-one"]')) {
      await act(() => resetOrbitals({ side: view.side, name: view.name }), '[data-dlg="close"]');
      ctx.offerUndo();
    }
  };
}

/** A change that redraws the whole card: the switch, a move, a reset. */
async function act(request, focus) {
  try {
    draw(await ctx.apply(request));
  } catch (error) {
    await ctx.refused(error);
    draw(ctx.state());
  }
  if (focus) $(focus, dlg)?.focus();
}

/** Send one field. The card is not redrawn, so what is being typed elsewhere stays. */
async function commit(el) {
  const key = el.dataset.key;
  const value = read(el);
  clear(el);
  let state;
  try {
    state = await ctx.apply(() => patchOrbital(view.side, view.name, { settings: { [key]: value } }));
  } catch (error) {
    // 422: the value; 409: the chain refuses it. Both belong under the field.
    if (error.status === 422 || error.status === 409) { fail(el, error.message); return; }
    closeDialog();
    await ctx.refused(error);
    return;
  }
  if (key === "name") view.name = value;
  const o = find(state, view.side, view.name);
  if (!o) { closeDialog(); return; }
  $("#card-title").textContent = o.name;
  $("#card-switch").setAttribute("aria-label", o.name);
  $("#card-fp").title = o.fingerprint;
  $("#card-fp").textContent = `fingerprint ${shortFp(o.fingerprint)} · changes with every setting and is recorded in each turn report`;
  const reset = $('[data-dlg="reset-one"]', dlg);
  if (reset) reset.disabled = !o.modified;
  for (const s of o.settings) $(`#${CSS.escape(fieldId(s.key))}`, dlg)?.closest(".field").classList.toggle("changed", s.changed);
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
      <p class="hint">Orbitals that need your own objects, such as clients or functions, are added in code.</p>`,
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
    body: `<form id="add-form" class="fields" novalidate>${nameField(kind.name)}${kind.settings.map((s) => field(s, s.default, names)).join("")}</form>
      <p class="field-err" id="add-err" role="alert" hidden></p>`,
    actions: `<button class="btn" type="button" data-dlg="back">Back</button><span class="spacer"></span><button class="btn" type="button" data-dlg="close">Cancel</button><button class="btn primary" type="submit" form="add-form">Add</button>`,
  });
  $$("textarea", dlg).forEach(grow);
  $("#dlg-body").oninput = (e) => { if (e.target.tagName === "TEXTAREA") grow(e.target); };
  $("#dlg-actions").onclick = (e) => { if (e.target.closest('[data-dlg="back"]')) showPicker(side, ctx); };
  ($(`#${CSS.escape(fieldId((kind.settings.find((s) => s.required) || {}).key || "name"))}`, dlg) || $("#f-name")).focus();
  $("#add-form").onsubmit = async (e) => {
    e.preventDefault();
    const settings = {};
    for (const el of $$("[data-key]", e.target)) { clear(el); settings[el.dataset.key] = read(el); }
    $("#add-err").hidden = true;
    try {
      await ctx.apply(() => addOrbital(side, kind.kind, settings));
    } catch (error) {
      const el = error.setting ? $(`#${CSS.escape(fieldId(error.setting))}`, dlg) : null;
      if (el) { fail(el, error.message); el.focus(); return; }
      $("#add-err").textContent = error.message;
      $("#add-err").hidden = false;
      return;
    }
    view = { side, name: settings.name };
    draw(ctx.state());
    ctx.toast(`${settings.name} added to the ${side} chain`);
  };
}

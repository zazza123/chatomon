import { $, $$, esc, rich } from "./util.js";

/**
 * Blocks: settings whose value has fields of its own, such as a connection or a service
 * description. The server says which forms a block accepts and which fields each type has;
 * this module draws them and reads the value back as JSON. It never decides what is valid.
 */

export const TRASH = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/></svg>`;

export const HELP = {
  env: "The name of an environment variable. Its value stays on the server.",
  lines: "One per line.",
};
// What is drawn as text has its value set once it is in the page: a text area drops a first
// line break written inside its tag, and a block is sent back whole, so every value must
// come back exactly as it arrived.
const values = new Map();
const drawn = new Map();
let counter = 0;

const modeOf = (value) => (value == null ? "" : typeof value === "string" ? "text" : value.type);
const modeLabel = (mode) => (mode === "" ? "None" : mode === "text" ? "One text" : mode);
const listed = (field) => field.type === "lines" || field.type === "items";
const blank = (type) => ({
  type: type.type,
  fields: Object.fromEntries(type.fields.map((f) => [f.key, listed(f) ? [] : f.default ?? null])),
});

function control(field, value, id) {
  const none = field.required ? "" : ' placeholder="None"';
  const attrs = `id="${id}" data-f="${esc(field.key)}" data-ft="${field.type}"`;
  const empty = `<option value=""${value == null ? " selected" : ""}>None</option>`;
  const option = (o, selected) => `<option${selected ? " selected" : ""}>${esc(o)}</option>`;
  if (field.type === "choice") return `<select ${attrs}>${field.optional ? empty : ""}${field.options.map((o) => option(o, o === value)).join("")}</select>`;
  if (field.type === "bool") return `<select ${attrs}>${field.optional ? empty : ""}${option("true", value === true)}${option("false", value === false)}</select>`;
  values.set(id, field.type === "lines" ? (value || []).join("\n") : value ?? "");
  if (field.type === "env") return `<input type="text" class="mono" ${attrs} aria-describedby="${id}-help" autocapitalize="none" autocomplete="off" spellcheck="false"${none}>`;
  if (field.type === "int" || field.type === "number") return `<input type="text" inputmode="${field.type === "int" ? "numeric" : "decimal"}" autocomplete="off" ${attrs}${none}>`;
  return `<textarea ${attrs} rows="${field.type === "lines" ? 2 : 1}"${HELP[field.type] ? ` aria-describedby="${id}-help"` : ""}${none}></textarea>`;
}

function entry(field, item, pid) {
  const id = `${pid}-${++counter}`;
  let body;
  if (typeof item === "string") {
    values.set(id, item);
    body = `<span class="item-kind">sentence</span><div class="field"><label class="sr" for="${id}">Sentence</label><textarea id="${id}" data-item-text rows="2"></textarea></div>`;
  } else {
    const type = field.types.find((known) => known.type === item.type);
    body = `<span class="item-kind">${esc(type.type)}</span>${type.fields.map((f) => `<div class="field"><label for="${id}-${esc(f.key)}">${esc(f.key)}</label>${control(f, item.fields[f.key], `${id}-${f.key}`)}</div>`).join("")}`;
  }
  return `<li class="item" data-item="${typeof item === "string" ? "text" : esc(item.type)}"><div class="item-main">${body}</div><button class="icon-btn sm danger" type="button" data-blk="remove" title="Remove">${TRASH}</button></li>`;
}

function row(field, value, id) {
  if (field.type === "items") {
    const items = Array.isArray(value) ? value : [];
    const adds = field.types.map((type) => `<button class="btn sm" type="button" data-blk="add" data-type="${esc(type.type)}">Add ${esc(type.type)}</button>`).join("");
    return `<div class="field wide" data-f="${esc(field.key)}" data-ft="items" data-pid="${id}"><span class="lbl" id="${id}-lbl">${esc(field.key)}</span>
      <ol class="items" aria-labelledby="${id}-lbl">${items.map((item) => entry(field, item, id)).join("")}</ol>
      <div class="item-add">${adds}${field.text ? `<button class="btn sm" type="button" data-blk="add" data-type="text">Add a sentence</button>` : ""}</div></div>`;
  }
  const wide = ["text", "env", "lines"].includes(field.type);
  const help = HELP[field.type] ? `<span class="help" id="${id}-help">${HELP[field.type]}</span>` : "";
  return `<div class="field${wide ? " wide" : ""}"><label for="${id}">${esc(field.key)}</label>${control(field, value, id)}${help}</div>`;
}

function body(setting, mode, value, pid) {
  if (mode === "") return `<p class="hint">Not given.</p>`;
  if (mode === "text") {
    values.set(`${pid}-text`, typeof value === "string" ? value : "");
    return `<div class="field wide"><label class="sr" for="${pid}-text">${esc(setting.key)}, as one text</label><textarea id="${pid}-text" data-blk-text rows="3"></textarea></div>`;
  }
  const type = setting.types.find((known) => known.type === mode);
  const held = value && typeof value === "object" && value.type === mode ? value : blank(type);
  return type.fields.map((f) => row(f, held.fields[f.key], `${pid}-${f.key}`)).join("");
}

/** The HTML of a block. Call `hydrate` once it is in the page. */
export function block(setting, value, pid, changed = false) {
  const modes = [...(setting.optional ? [""] : []), ...(setting.text ? ["text"] : []), ...setting.types.map((type) => type.type)];
  const mode = modes.includes(modeOf(value)) ? modeOf(value) : modes[0];
  const single = modes.length === 1;
  drawn.set(pid, { setting, value });
  const pick = single ? "" : `<div class="field"><label for="${pid}-as">given as</label><select id="${pid}-as" data-blk="mode">${modes.map((m) => `<option value="${esc(m)}"${m === mode ? " selected" : ""}>${esc(modeLabel(m))}</option>`).join("")}</select></div>`;
  return `<fieldset class="block${changed ? " changed" : ""}" data-block="${esc(setting.key)}" data-pid="${pid}" data-only="${single ? esc(modes[0]) : ""}" aria-describedby="${pid}-err">
    <legend>${esc(setting.key)}${single ? ` · ${esc(modes[0])}` : ""}</legend>
    <div class="block-body">${pick}<div class="fields" data-blk-body>${body(setting, mode, value, pid)}</div>
    <p class="block-state" role="status" hidden>Not applied yet: it is applied when you leave the block.</p>
    <p class="field-err" id="${pid}-err" role="alert" hidden></p></div></fieldset>`;
}

const grow = (el) => { el.style.height = "auto"; el.style.height = `${el.scrollHeight + 2}px`; };

function number(list) {
  const name = list.closest("[data-f]").dataset.f;
  $$(":scope > .item > [data-blk='remove']", list).forEach((button, i) => button.setAttribute("aria-label", `Remove entry ${i + 1} of ${name}`));
}

/** Give the blocks drawn under `root` their values, and remember what each one accepts and holds. */
export function hydrate(root) {
  for (const fs of $$("fieldset.block", root)) {
    if (!drawn.has(fs.dataset.pid)) continue;
    fs._setting = drawn.get(fs.dataset.pid).setting;
    keep(fs, drawn.get(fs.dataset.pid).value);
    drawn.delete(fs.dataset.pid);
  }
  for (const [id, value] of values) {
    const el = root.querySelector(`#${CSS.escape(id)}`);
    if (el) el.value = value;
  }
  values.clear();
  $$("fieldset.block textarea", root).forEach(grow);
  $$("fieldset.block .items", root).forEach(number);
}

function readControl(el) {
  const type = el.dataset.ft;
  if (type === "lines") return el.value.split("\n").map((line) => line.trim()).filter(Boolean);
  if (el.tagName === "SELECT") return el.value === "" ? null : type === "bool" ? el.value === "true" : el.value;
  if (el.value.trim() === "") return null;
  if (type !== "int" && type !== "number") return el.value;
  const n = Number(el.value.trim());
  return Number.isFinite(n) ? n : el.value;
}

/** The fields drawn directly under `root`: those of an entry belong to the entry. */
function readFields(root, inEntry) {
  const fields = {};
  for (const el of $$("[data-f]", root)) {
    if (Boolean(el.closest(".item")) !== inEntry) continue;
    fields[el.dataset.f] = el.dataset.ft === "items"
      ? $$(":scope > .items > .item", el).map((li) => (li.dataset.item === "text" ? $("[data-item-text]", li).value : { type: li.dataset.item, fields: readFields(li, true) })).filter((item) => typeof item !== "string" || item.trim() !== "")
      : readControl(el);
  }
  return fields;
}

/** Remember the value a block has on the server: choosing its form again shows it. */
export function keep(fs, value) {
  fs._value = value;
}

/** The JSON value of a block: none, one text, or a type with its fields. */
export function read(fs) {
  const pick = $("[data-blk='mode']", fs);
  const mode = pick ? pick.value : fs.dataset.only;
  if (mode === "") return null;
  if (mode === "text") return $("[data-blk-text]", fs).value;
  return { type: mode, fields: readFields(fs, false) };
}

/** Show a refusal under a block, or clear it. */
export function mark(fs, message) {
  const box = $(":scope > .block-body > .field-err", fs);
  box.hidden = !message;
  box.innerHTML = message ? rich(message) : "";
  fs.toggleAttribute("aria-invalid", Boolean(message));
}

/**
 * Handle what happens inside a block without leaving it: another form is chosen, an entry
 * is added or removed. Returns the block that changed, or null when the event is not one of these.
 */
export function handle(e) {
  const el = e.target.closest("[data-blk]");
  const fs = el && el.closest("fieldset.block");
  if (!fs) return null;
  const kind = el.dataset.blk;
  if (kind === "mode" && e.type === "change") {
    // The form the block has comes back with its value; any other form starts empty.
    $("[data-blk-body]", fs).innerHTML = body(fs._setting, el.value, fs._value, fs.dataset.pid);
    hydrate(fs);
    return fs;
  }
  if (e.type !== "click" || kind === "mode") return null;
  const group = el.closest("[data-ft='items']");
  const list = $(":scope > .items", group);
  if (kind === "add") {
    const type = fs._setting.types.find((known) => known.type === fs.dataset.only || known.type === $("[data-blk='mode']", fs)?.value);
    const field = type.fields.find((f) => f.key === group.dataset.f);
    const added = el.dataset.type === "text" ? "" : blank(field.types.find((known) => known.type === el.dataset.type));
    list.insertAdjacentHTML("beforeend", entry(field, added, group.dataset.pid));
    hydrate(fs);
    $(":scope > .item:last-child textarea, :scope > .item:last-child input, :scope > .item:last-child select", list).focus();
  } else {
    // The focus goes to the list's Add button before the entry goes, so it stays in the block.
    $(":scope > .item-add > [data-blk='add']", group).focus();
    el.closest(".item").remove();
    number(list);
  }
  return fs;
}

/** Whether Enter in this control starts a new line instead of applying. */
export const breaksLine = (el) => el.dataset.ft === "lines";

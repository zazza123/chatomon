import { $, esc, plural, rich } from "./util.js";
import { getCode, getPresetText, importPreset } from "./api.js";
import { closeDialog, openDialog } from "./cards.js";

const dlg = $("#dlg");
const list = (items) => `<ul>${items.map((item) => `<li>${rich(item)}</li>`).join("")}</ul>`;
const CLOSE = `<button class="btn" type="button" data-dlg="close">Close</button>`;
const COPY = `<button class="btn primary" type="button" data-dlg="copy">Copy</button>`;

/** Colour Python or JSON for reading. The text itself comes from the server. */
function highlight(code, lang) {
  const re = lang === "json"
    ? /("(?:[^"\\]|\\.)*")(\s*:)?|\b(true|false|null)\b|(-?\b\d+(?:\.\d+)?\b)/g
    : /(#[^\n]*)|("(?:[^"\\\n]|\\.)*")|\b(from|import|None|True|False)\b|\b(\d+(?:\.\d+)?)\b/g;
  let out = "";
  let last = 0;
  let m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index));
    if (lang === "json") {
      if (m[1]) out += m[2] ? `<span class="k">${esc(m[1])}</span>${esc(m[2])}` : `<span class="s">${esc(m[1])}</span>`;
      else out += `<span class="n">${esc(m[3] || m[4])}</span>`;
    } else if (m[1]) out += `<span class="c">${esc(m[1])}</span>`;
    else if (m[2]) out += `<span class="s">${esc(m[2])}</span>`;
    else if (m[3]) out += `<span class="k">${esc(m[3])}</span>`;
    else out += `<span class="n">${esc(m[4])}</span>`;
    last = re.lastIndex;
  }
  return out + esc(code.slice(last));
}

async function copy(text, toast) {
  try {
    await navigator.clipboard.writeText(text);
    toast("Copied");
  } catch {
    // No clipboard access: select the text so one key press copies it.
    const code = $("pre code", dlg);
    if (code) {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    }
    toast("Selected: press Ctrl+C or ⌘C to copy");
  }
}

const onCopy = (text, ctx) => { $("#dlg-actions").onclick = (e) => { if (e.target.closest('[data-dlg="copy"]')) copy(text, ctx.toast); }; };

export async function showCode(ctx) {
  const data = await getCode();
  const count = data.placeholders.length;
  const changes = data.changes.length
    ? `<p class="note">Studio changes: ${esc(data.changes.join(", "))}.</p>`
    : `<p class="note">Same configuration as your code.</p>`;
  const marks = count
    ? `<div class="note warn">${plural(count, "part")} could not be written as code and ${count === 1 ? "is" : "are"} marked <code>...</code>:${list(data.placeholders)}</div>`
    : "";
  openDialog({
    size: "l",
    title: "Copy as code",
    sub: `The orbitals of <code>${esc(ctx.state().atom.source)}</code> as they are in the studio`,
    body: `${changes}${marks}${data.notes.map((note) => `<p class="note">${rich(note)}</p>`).join("")}${data.code ? `<pre class="code"><code>${highlight(data.code, "python")}</code></pre>` : ""}`,
    actions: `<span class="hint spacer">Paste the lists over the ones in your atom.</span>${CLOSE}${data.code ? COPY : ""}`,
  });
  onCopy(data.code, ctx);
}

export async function showExport(ctx) {
  const text = await getPresetText();
  openDialog({
    size: "l",
    title: "Export preset",
    sub: "Which orbitals are on, their order, and the settings you changed",
    body: `<p class="note">A preset is a layer over your code: it changes the orbitals your code defines and can add built-in ones. It holds no clients and no credentials.</p>
      <pre class="code"><code>${highlight(text, "json")}</code></pre>`,
    actions: `<span class="spacer"></span>${CLOSE}<a class="btn" href="/api/preset?download=1" download>Download</a>${COPY}`,
  });
  onCopy(text, ctx);
}

export function showImport(ctx) {
  openDialog({
    size: "l",
    title: "Import preset",
    sub: "Choose a preset file, or paste its text",
    body: `<p class="note">The preset replaces the changes of this session; Undo brings them back. Orbitals from your code keep their objects.</p>
      <input type="file" id="import-file" accept=".json,application/json" aria-label="Preset file">
      <label class="sr" for="import-text">Preset JSON</label>
      <textarea id="import-text" spellcheck="false" placeholder='{ "chatomon_studio_preset": 1, … }'></textarea>
      <div class="note err" id="import-problems" role="alert" hidden></div>`,
    actions: `<span class="spacer"></span><button class="btn" type="button" data-dlg="close">Cancel</button><button class="btn primary" type="button" data-dlg="apply">Apply preset</button>`,
  });
  $("#import-file").onchange = async (e) => {
    const file = e.target.files[0];
    if (file) $("#import-text").value = await file.text();
  };
  $("#dlg-actions").onclick = async (e) => {
    if (!e.target.closest('[data-dlg="apply"]')) return;
    const box = $("#import-problems");
    box.hidden = true;
    let result;
    try {
      result = await importPreset($("#import-text").value);
    } catch (error) {
      box.innerHTML = error.problems ? `The preset was not applied:${list(error.problems)}` : rich(error.message);
      box.hidden = false;
      return;
    }
    ctx.refresh(result.state);
    if (!result.warnings.length) { closeDialog(); ctx.offerUndo(); return; }
    openDialog({
      title: "Preset applied",
      sub: plural(result.warnings.length, "note"),
      body: `<div class="note warn">The session is not exactly the one that was exported:${list(result.warnings)}</div>`,
      actions: `<button class="btn" type="button" data-dlg="undo">Undo</button><span class="spacer"></span><button class="btn primary" type="button" data-dlg="close">Done</button>`,
    });
    $("#dlg-actions").onclick = (event) => { if (event.target.closest('[data-dlg="undo"]')) { closeDialog(); ctx.undo(); } };
  };
}

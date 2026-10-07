import { $, esc, plural, rich } from "./util.js";
import { getCode, getPresetText, getTest, importPreset, writeTest } from "./api.js";
import { closeDialog, openDialog } from "./cards.js";

const dlg = $("#dlg");
const list = (items) => `<ul>${items.map((item) => `<li>${rich(item)}</li>`).join("")}</ul>`;
const CLOSE = `<button class="btn" type="button" data-dlg="close">Close</button>`;
const COPY = `<button class="btn primary" type="button" data-dlg="copy">Copy</button>`;

/** Colour Python or JSON for reading. The text itself comes from the server. */
function highlight(code, lang) {
  const re = lang === "json"
    ? /("(?:[^"\\]|\\.)*")(\s*:)?|\b(true|false|null)\b|(-?\b\d+(?:\.\d+)?\b)/g
    : /(#[^\n]*)|("""[\s\S]*?"""|"(?:[^"\\\n]|\\.)*")|(@[\w.]+)|\b(from|import|as|class|def|async|await|return|yield|raise|assert|for|in|if|None|True|False)\b|\b(\d+(?:\.\d+)?)\b/g;
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
    else if (m[3]) out += `<span class="d">${esc(m[3])}</span>`;
    else if (m[4]) out += `<span class="k">${esc(m[4])}</span>`;
    else out += `<span class="n">${esc(m[5])}</span>`;
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

export async function showCode(ctx, { onBack = null, onClose = () => {} } = {}) {
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
    onClose,
    title: "Copy as code",
    sub: `The orbitals of <code>${esc(ctx.state().atom.source)}</code> as they are in the studio`,
    body: `${changes}${marks}${data.notes.map((note) => `<p class="note">${rich(note)}</p>`).join("")}${data.code ? `<pre class="code"><code>${highlight(data.code, "python")}</code></pre>` : ""}`,
    actions: `${onBack ? `<button class="btn" type="button" data-dlg="back">Back to the test</button>` : ""}<span class="hint spacer">Paste the lists over the ones in your atom.</span>${CLOSE}${data.code ? COPY : ""}`,
  });
  $("#dlg-actions").onclick = (e) => {
    if (e.target.closest('[data-dlg="copy"]')) copy(data.code, ctx.toast);
    if (onBack && e.target.closest('[data-dlg="back"]')) onBack();
  };
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
    const { connections, warnings } = result;
    if (!connections.length && !warnings.length) { closeDialog(); ctx.offerUndo(); return; }
    const differs = warnings.length ? `<div class="note warn" id="preset-warnings">The session is not exactly the one that was exported:${list(warnings)}</div>` : "";
    openDialog({
      title: "Preset applied",
      sub: plural(connections.length + warnings.length, "note"),
      body: `${connects(connections)}${differs}`,
      actions: `<button class="btn" type="button" data-dlg="undo">Undo</button><span class="spacer"></span><button class="btn primary" type="button" data-dlg="close">Done</button>`,
    });
    $("#dlg-actions").onclick = (event) => { if (event.target.closest('[data-dlg="undo"]')) { closeDialog(); ctx.undo(); } };
  };
}

/** Where an orbital connects is a note of its own: it is not about what differs. */
const connects = (connections) => (connections.length ? `<div class="note warn" id="preset-connections">This preset decides where an orbital connects:${list(connections)}</div>` : "");

/** Say, when the page opens, where the preset the studio started with makes orbitals connect. */
export function showStarted(connections, onClose) {
  openDialog({
    title: "Started with a preset",
    sub: plural(connections.length, "note"),
    body: connects(connections),
    actions: `<span class="spacer"></span><button class="btn primary" type="button" data-dlg="close">Done</button>`,
    onClose,
  });
}

/** Save a turn as a test. The code comes from the server, which is also what writes the file. */
export async function showTest(n, ctx, { onBack, onClose, alone = false, path = null, focus = null }) {
  const data = await getTest(n, alone);
  const warnings = data.warnings.length
    ? `<div class="note warn" role="alert">${data.warnings.map((warning) => `<p>${rich(warning)}</p>`).join("")}<button class="btn sm" type="button" data-dlg="code">Copy as code</button></div>`
    : "";
  const earlier = data.earlier
    ? `<label class="check"><input type="checkbox" id="test-earlier"${alone ? "" : " checked"}> Replay the ${plural(data.earlier, "turn")} before it</label>`
    : "";
  const file = data.writable
    ? `<div class="field"><label for="test-path">File, inside <code>${esc(data.root)}</code></label>
        <input type="text" id="test-path" value="${esc(path ?? data.path)}" spellcheck="false" autocomplete="off" aria-describedby="test-path-msg">
        <p class="field-err" id="test-path-msg" role="alert" hidden></p></div>`
    : "";
  openDialog({
    size: "l",
    onClose,
    title: `Save turn ${n} as a test`,
    sub: data.integration ? "Calls real models: marked <code>integration</code>" : "Runs offline: no model is called",
    body: `${warnings}${data.notes.map((note) => `<p class="note">${rich(note)}</p>`).join("")}${earlier}${file}
      <pre class="code"><code>${highlight(data.code, "python")}</code></pre>
      <p class="note" id="test-done" role="status" tabindex="-1" hidden></p>`,
    actions: `<button class="btn" type="button" data-dlg="back">Back to the report</button><span class="spacer"></span>${data.writable ? `<button class="btn" type="button" data-dlg="copy">Copy</button><button class="btn primary" type="button" data-dlg="write">Write file</button>` : COPY}`,
  });
  // The card was drawn again: the focus goes back to what was used.
  if (focus) $(focus, dlg)?.focus();
  const typed = () => $("#test-path")?.value ?? null;

  async function write(overwrite) {
    const field = $("#test-path");
    const message = $("#test-path-msg");
    const done = $("#test-done");
    message.hidden = true;
    done.hidden = true;
    field.removeAttribute("aria-invalid");
    try {
      // The turn is named by the id the test came with: its number alone can be another turn by now.
      const saved = await writeTest(n, data.turn, field.value, alone, overwrite);
      done.innerHTML = rich(`Saved as \`${saved.path}\`. Run it with \`${saved.command}\`.`);
      done.hidden = false;
      ctx.toast("Test saved");
      // The Overwrite button that had the focus is hidden again: the result takes it.
      if (overwrite) done.focus();
    } catch (error) {
      // A file of the studio that exists is replaced only on a second, explicit click.
      const own = error.type === "Exists";
      field.setAttribute("aria-invalid", "true");
      message.innerHTML = rich(error.message) + (own ? ` <button class="btn sm" type="button" data-dlg="overwrite">Overwrite</button>` : "");
      message.hidden = false;
      (own ? $("button", message) : field).focus();
    }
  }

  $("#dlg-body").onchange = (e) => {
    if (e.target.id !== "test-earlier") return;
    const box = e.target;
    showTest(n, ctx, { onBack, onClose, alone: !box.checked, path: typed(), focus: "#test-earlier" }).catch((error) => {
      // The card still shows the other test, which is the one that would be written.
      box.checked = !box.checked;
      ctx.refused(error);
    });
  };
  $("#dlg-body").onkeydown = (e) => {
    if (e.key === "Enter" && e.target.id === "test-path") { e.preventDefault(); write(false); }
  };
  $("#dlg-body").onclick = (e) => {
    if (e.target.closest('[data-dlg="overwrite"]')) write(true);
    if (e.target.closest('[data-dlg="code"]')) {
      const again = { onBack, onClose, alone, path: typed(), focus: '[data-dlg="code"]' };
      showCode(ctx, { onClose, onBack: () => showTest(n, ctx, again).catch(ctx.refused) }).catch(ctx.refused);
    }
  };
  $("#dlg-actions").onclick = (e) => {
    if (e.target.closest('[data-dlg="back"]')) onBack(n);
    if (e.target.closest('[data-dlg="copy"]')) copy(data.code, ctx.toast);
    if (e.target.closest('[data-dlg="write"]')) write(false);
  };
}

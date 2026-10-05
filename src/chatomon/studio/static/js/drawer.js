import { $, $$, esc, plural, storage } from "./util.js";

const wide = window.matchMedia("(min-width: 1000px)");
const GRIP = `<svg viewBox="0 0 8 14" aria-hidden="true"><g fill="currentColor"><circle cx="2" cy="2" r="1.3"/><circle cx="6" cy="2" r="1.3"/><circle cx="2" cy="7" r="1.3"/><circle cx="6" cy="7" r="1.3"/><circle cx="2" cy="12" r="1.3"/><circle cx="6" cy="12" r="1.3"/></g></svg>`;
const TRASH = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/></svg>`;

export function createDrawer({ onToggle, onOrbital, onAtom, onMove, onDelete, onAdd, onResetAll, onCode, onExport, onImport, isBusy }) {
  const app = $("#app");
  let open = wide.matches ? storage.get("chatomon-studio-drawer") !== "closed" : false;

  function sync() {
    app.dataset.drawer = open ? "open" : "closed";
    $("#drawer-btn").setAttribute("aria-expanded", String(open));
    $("#drawer").inert = !open || isBusy();
    $("#scrim").hidden = !(open && !wide.matches);
  }
  function set(value) {
    open = value;
    sync();
    if (wide.matches) storage.set("chatomon-studio-drawer", open ? "open" : "closed");
  }
  $("#drawer-btn").addEventListener("click", () => set(!open));
  $("#scrim").addEventListener("click", () => set(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && open && !wide.matches && !$("#dlg").open) set(false); });
  wide.addEventListener("change", () => { open = wide.matches; sync(); });

  const drawer = $("#drawer");
  drawer.addEventListener("click", (e) => {
    const b = e.target.closest("[data-act]");
    if (!b || b.tagName === "INPUT") return;
    const li = b.closest(".orb");
    const side = li ? li.dataset.side : b.dataset.side;
    switch (b.dataset.act) {
      case "card": onOrbital(side, li.dataset.name); break;
      case "delete": onDelete(side, li.dataset.name); break;
      case "add": onAdd(side); break;
      case "atom": onAtom(); break;
      case "collapse": set(false); $("#drawer-btn").focus(); break;
      case "reset-all": onResetAll(); break;
      case "copy-code": onCode(); break;
      case "export": onExport(); break;
      case "import": onImport(); break;
    }
  });
  drawer.addEventListener("change", (e) => {
    const el = e.target;
    if (el.dataset.act === "enable") onToggle(el.dataset.side, el.dataset.name, el.checked);
  });

  // Dragging a row moves its orbital inside its own chain; the card has buttons for the same.
  let drag = null;
  const clearDrop = () => $$(".drop-before, .drop-after", drawer).forEach((x) => x.classList.remove("drop-before", "drop-after"));
  drawer.addEventListener("dragstart", (e) => {
    const li = e.target.closest ? e.target.closest(".orb") : null;
    if (!li) return;
    drag = { side: li.dataset.side, name: li.dataset.name, index: Number(li.dataset.index) };
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", li.dataset.name);
    li.classList.add("dragging");
  });
  drawer.addEventListener("dragover", (e) => {
    const li = e.target.closest ? e.target.closest(".orb") : null;
    if (!drag || !li || li.dataset.side !== drag.side) return;
    e.preventDefault();
    clearDrop();
    const box = li.getBoundingClientRect();
    li.classList.add(e.clientY < box.top + box.height / 2 ? "drop-before" : "drop-after");
  });
  drawer.addEventListener("drop", (e) => {
    const li = e.target.closest ? e.target.closest(".orb") : null;
    if (!drag || !li || li.dataset.side !== drag.side) return;
    e.preventDefault();
    const target = Number(li.dataset.index);
    let position = li.classList.contains("drop-after") ? target + 1 : target;
    if (drag.index < position) position -= 1; // the row leaves its own place first
    const moved = drag;
    drag = null;
    clearDrop();
    if (position !== moved.index) onMove(moved.side, moved.name, position);
  });
  drawer.addEventListener("dragend", () => {
    drag = null;
    clearDrop();
    $$(".dragging", drawer).forEach((x) => x.classList.remove("dragging"));
  });

  function row(o, index) {
    const added = o.origin === "studio";
    const name = esc(o.name);
    return `<li class="orb${o.enabled ? "" : " off"}" data-side="${o.side}" data-name="${name}" data-index="${index}">
      <div class="orb-row" draggable="true">
        <span class="grip" title="Drag to reorder" aria-hidden="true">${GRIP}</span>
        <span class="orb-glyph" aria-hidden="true"></span>
        <button class="orb-main" type="button" data-act="card" title="Open ${name}"><span class="orb-name">${name}</span><span class="orb-kind">${esc(o.kind)}${added ? " · <em>added in the studio</em>" : ""}</span></button>
        <span class="mod-dot"${o.modified ? "" : " hidden"} title="Changed in the studio"><span class="sr">changed in the studio</span></span>
        ${added ? `<button class="icon-btn sm danger" type="button" data-act="delete" aria-label="Delete ${name}" title="Delete">${TRASH}</button>` : ""}
        <label class="switch" title="${o.enabled ? "Turn off" : "Turn on"}"><input type="checkbox" role="switch" data-act="enable" data-side="${o.side}" data-name="${name}"${o.enabled ? " checked" : ""} aria-label="${name}"><span></span></label>
      </div></li>`;
  }

  sync();

  return {
    sync,
    render(state) {
      $("#atom-name").textContent = state.atom.name;
      $("#atom-source").textContent = state.atom.source;
      $("#nucleus-kind").textContent = `${state.atom.connector} · ${state.atom.model}`;
      for (const side of ["input", "output"]) {
        const list = state.orbitals[side];
        const on = list.filter((o) => o.enabled).length;
        $(`#count-${side}`).textContent = on === list.length ? `${on} on` : `${on} of ${list.length} on`;
        $(`#list-${side}`).innerHTML = list.length ? list.map(row).join("") : `<li class="muted">None</li>`;
      }
      const changes = state.changes;
      const count = `${plural(changes.length, "change")} vs code`;
      $("#changes").classList.toggle("dirty", changes.length > 0);
      $("#changes-text").textContent = changes.length ? count : `Same as ${state.atom.source}`;
      $("#changes-list").innerHTML = changes.map((change) => `<li>${esc(change.text)}</li>`).join("");
      if (!changes.length) $("#changes-details").open = false;
      $("#reset-all").hidden = !changes.length;
      $("#drawer-badge").hidden = !changes.length;
      $("#drawer-btn").title = changes.length ? `Show or hide the atom (${count})` : "Show or hide the atom";
      sync();
    },
  };
}

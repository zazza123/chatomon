import { $, esc, storage } from "./util.js";

const wide = window.matchMedia("(min-width: 1000px)");

export function createDrawer({ onToggle, onOrbital, onAtom, isBusy }) {
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
    if (b.dataset.act === "card") onOrbital(b.dataset.side, b.dataset.name);
    if (b.dataset.act === "atom") onAtom();
    if (b.dataset.act === "collapse") { set(false); $("#drawer-btn").focus(); }
  });
  drawer.addEventListener("change", (e) => {
    const el = e.target;
    if (el.dataset.act === "enable") onToggle(el.dataset.side, el.dataset.name, el.checked);
  });

  function row(o) {
    return `<li class="orb${o.enabled ? "" : " off"}">
      <div class="orb-row">
        <span class="orb-glyph" aria-hidden="true"></span>
        <button class="orb-main" type="button" data-act="card" data-side="${o.side}" data-name="${esc(o.name)}" title="Open ${esc(o.name)}"><span class="orb-name">${esc(o.name)}</span><span class="orb-kind">${esc(o.kind)}</span></button>
        <label class="switch" title="${o.enabled ? "Turn off" : "Turn on"}"><input type="checkbox" role="switch" data-act="enable" data-side="${o.side}" data-name="${esc(o.name)}"${o.enabled ? " checked" : ""} aria-label="${esc(o.name)}"><span></span></label>
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
      sync();
    },
  };
}

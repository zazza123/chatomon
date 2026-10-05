import { $, reduceMotion, storage } from "./util.js";
import { deleteOrbital, getState, patchOrbital, resetConversation, resetOrbitals, sendTurn, undoOrbitals } from "./api.js";
import { createStage } from "./stage.js";
import { renderTurns } from "./turns.js";
import { createDrawer } from "./drawer.js";
import { showAtom, showNotice, showReport } from "./cards.js";
import { showOrbital, showPicker } from "./orbital-card.js";
import { showCode, showExport, showImport } from "./export-cards.js";

let toastTimer = 0;
let toastFn = null;
function toast(text, actionLabel, fn) {
  const t = $("#toast");
  $("#toast-text").textContent = text;
  const act = $("#toast-act");
  act.hidden = !actionLabel;
  act.textContent = actionLabel || "";
  toastFn = fn || null;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), actionLabel ? 5000 : 2200);
}
$("#toast-act").addEventListener("click", () => {
  const fn = toastFn;
  toastFn = null;
  $("#toast").classList.remove("show");
  if (fn) fn();
});

const THEMES = ["system", "light", "dark"];
const THEME_ICONS = {
  system: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16Z" fill="currentColor"/></svg>`,
  light: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>`,
  dark: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z"/></svg>`,
};
function applyTheme(mode) {
  const root = document.documentElement;
  if (mode === "system") delete root.dataset.theme; else root.dataset.theme = mode;
  const btn = $("#theme");
  btn.innerHTML = THEME_ICONS[mode];
  btn.title = `Theme: ${mode}`;
  btn.setAttribute("aria-label", `Theme: ${mode}. Change theme`);
  btn.dataset.mode = mode;
}
$("#theme").addEventListener("click", () => {
  const next = THEMES[(THEMES.indexOf($("#theme").dataset.mode) + 1) % THEMES.length];
  applyTheme(next);
  storage.set("chatomon-studio-theme", next);
});


const ui = { state: null, pending: null, busy: false, inspected: null };
const play = $("#play");
const SYMBOL = $(".brand .sym").outerHTML;
const scrollToEnd = () => play.scrollTo({ top: play.scrollHeight, behavior: reduceMotion ? "auto" : "smooth" });

function setCaption(title, sub = "", warn = false) {
  $("#phase-title").textContent = title;
  $("#phase-sub").textContent = sub;
  $("#dock-title").textContent = title;
  $("#dock-sub").textContent = sub;
  $(".phase").classList.toggle("warn", warn);
  $("#dock").classList.toggle("warn", warn);
}

/** Send a change to the orbitals: the page shows the new state, which is returned. */
async function apply(request) {
  const state = await request();
  refresh(state);
  return state;
}

/** A change was refused, or the page is out of date: say why and show the state as it is. */
async function refused(error) {
  toast(error.message);
  try { refresh(await getState()); } catch { lostStudio(); }
}

async function change(request, focus) {
  try { await apply(request); } catch (error) { await refused(error); return false; }
  // The drawer was drawn again: give the focus back to what was used.
  if (focus) document.querySelector(focus)?.focus();
  return true;
}

function offerUndo() {
  if (ui.state.undo) toast(ui.state.undo, "Undo", () => change(undoOrbitals));
}

const rowOf = (side, name) => `#drawer li.orb[data-side="${side}"][data-name="${CSS.escape(name)}"]`;
const toggle = (side, name, enabled) => change(() => patchOrbital(side, name, { enabled }), `${rowOf(side, name)} input`);
const move = (side, name, position) => change(() => patchOrbital(side, name, { position }), `${rowOf(side, name)} .orb-main`);
async function remove(side, name) { if (await change(() => deleteOrbital(side, name))) offerUndo(); }
async function resetAll() { if (await change(() => resetOrbitals())) offerUndo(); }

const cards = { state: () => ui.state, apply, refused, refresh, toast, offerUndo, undo: () => change(undoOrbitals), remove };

const openOrbital = (side, name) => showOrbital(side, name, cards);

const stage = createStage({
  svgs: [$("#atom"), $("#dock-atom")],
  setCaption,
  onOrbital: openOrbital,
  onAtom: () => showAtom(ui.state),
  onFinished(turn) {
    ui.state.turns.push(turn);
    ui.pending = null;
    setBusy(false);
    drawTurns();
    scrollToEnd();
  },
});
const drawer = createDrawer({
  onToggle: toggle,
  onOrbital: openOrbital,
  onAtom: () => showAtom(ui.state),
  onMove: move,
  onDelete: remove,
  onAdd: (side) => showPicker(side, cards),
  onResetAll: resetAll,
  onCode: () => showCode(cards).catch(refused),
  onExport: () => showExport(cards).catch(refused),
  onImport: () => showImport(cards),
  isBusy: () => ui.busy,
});

function drawTurns() {
  renderTurns($("#log"), ui.state.turns, {
    pending: ui.pending, busy: ui.busy, inspected: ui.inspected, symbol: SYMBOL,
    placeholder: ui.state.atom.blocked_placeholder,
  });
  const n = ui.state.turns.length;
  $("#turn-label").textContent = n ? `${n} turn${n === 1 ? "" : "s"} · next: turn ${n + 1}` : "Next: turn 1";
}

function setBusy(value) {
  ui.busy = value;
  $("#send").disabled = value;
  $("#new-conv").disabled = value;
  drawer.sync();
}

const LOCAL_NOTE = 'The studio runs your atom with capture_content="full": it shows message text, guidance and judge explanations.';

function refresh(state) {
  ui.state = state;
  const local = state.studio.local !== false;
  $("#chip-scope").textContent = local ? "\u00a0· local only" : "\u00a0· network address";
  $("#content-chip").title = local
    ? `${LOCAL_NOTE} It only listens on this machine.`
    : `${LOCAL_NOTE} It listens on a network address: anyone who reaches it with the link can read this conversation.`;
  drawer.render(state);
  stage.setOrbitals(state.orbitals, state.atom);
  drawTurns();
}

/** The studio stopped answering: free the page and say what to do. */
function lostStudio() {
  ui.pending = null;
  setBusy(false);
  drawTurns();
  toast("The studio is not answering. Start it again and open the link it prints.");
}

async function send(text) {
  text = text.trim();
  if (!text || ui.busy) return;
  ui.pending = { n: ui.state.turns.length + 1, message: text };
  setBusy(true);
  drawTurns();
  scrollToEnd();
  let finished = false;
  let failure = null;
  try {
    await sendTurn(text, (event) => {
      if (event.type === "turn_finished") finished = true;
      stage.push(event);
    });
  } catch (error) {
    failure = error;
  }
  // The stream ended without the turn's last event (a refused request or a lost connection):
  // ask the server where things stand.
  if (!finished) {
    ui.pending = null;
    try {
      await settle();
      if (failure) toast(failure.message);
    } catch {
      lostStudio();
    }
  }
}

/** Show the server's state, waiting for a turn that is still running. */
async function settle() {
  refresh(await getState());
  let waited = false;
  while (ui.state.busy) {
    waited = true;
    setBusy(true);
    ui.pending = ui.state.running;
    drawTurns();
    stage.showRunning();
    await new Promise((resolve) => setTimeout(resolve, 1000));
    refresh(await getState());
  }
  ui.pending = null;
  setBusy(false);
  drawTurns();
  if (waited) {
    stage.release();
    scrollToEnd();
  }
}

function inspect(n) {
  const turn = ui.state.turns[n - 1];
  if (!turn) return;
  ui.inspected = n;
  drawTurns();
  showReport(turn, {
    total: ui.state.turns.length,
    busy: ui.busy,
    onNav: inspect,
    onRerun: send,
    onClose: () => {
      const last = ui.inspected;
      ui.inspected = null;
      drawTurns();
      // The turns were drawn again: give the focus back to the lens that opened the report.
      document.querySelector(`.lens[data-n="${last}"]`)?.focus();
    },
  });
}

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  if (ui.busy) return; // Enter while a turn runs: keep what was typed
  const box = $("#msg");
  const text = box.value;
  box.value = "";
  autoSize();
  send(text);
});
$("#msg").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#composer").requestSubmit(); }
});
function autoSize() { const box = $("#msg"); box.style.height = "auto"; box.style.height = `${Math.min(box.scrollHeight + 2, 140)}px`; }
$("#msg").addEventListener("input", autoSize);
$("#log").addEventListener("click", (e) => {
  const b = e.target.closest("[data-act]");
  if (!b) return;
  const n = Number(b.dataset.n);
  if (b.dataset.act === "inspect") inspect(n);
  if (b.dataset.act === "rerun" && ui.state.turns[n - 1]) send(ui.state.turns[n - 1].message);
});
$("#new-conv").addEventListener("click", async () => {
  try { refresh(await resetConversation()); toast("New conversation"); } catch (error) { toast(error.message); }
});
for (const [id, value] of [["#sp-slow", true], ["#sp-real", false]]) {
  $(id).addEventListener("click", () => {
    stage.setSlow(value);
    $("#sp-slow").setAttribute("aria-pressed", String(value));
    $("#sp-real").setAttribute("aria-pressed", String(!value));
  });
}
new IntersectionObserver(([entry]) => { $("#dock").hidden = entry.isIntersecting; }, { root: play, threshold: 0.3 }).observe($("#stage"));
$("#dock").addEventListener("click", () => play.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" }));
$("#addr").textContent = location.host;
$("#content-chip").addEventListener("click", () => showNotice(ui.state ? ui.state.studio.local !== false : true));

async function boot() {
  applyTheme(["system", "light", "dark"].includes(storage.get("chatomon-studio-theme")) ? storage.get("chatomon-studio-theme") : "system");
  // A turn may be running from before a reload: settle() waits for it, then shows it.
  await settle();
  requestAnimationFrame(() => requestAnimationFrame(() => $("#app").classList.add("ready")));
}
boot().catch(lostStudio);

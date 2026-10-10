import { $, esc, fmtMs, wait, plural } from "./util.js";

export const POSES = ["rest", "listen", "think", "review", "block", "block-output", "reply", "idle"];

/* Drawing: the same geometry as the brand's mascot poses. */

function pt(r, deg) { const a = (deg * Math.PI) / 180; return [100 + r * Math.cos(a), 108 + r * Math.sin(a)]; }
function arc(r, a0, a1) {
  const [x0, y0] = pt(r, a0);
  const [x1, y1] = pt(r, a1);
  return `M${x0.toFixed(1)} ${y0.toFixed(1)} A${r} ${r} 0 ${Math.abs(a1 - a0) > 180 ? 1 : 0} ${a1 > a0 ? 1 : 0} ${x1.toFixed(1)} ${y1.toFixed(1)}`;
}
// Input dashes run from top-left down the left side; output dashes from top-right down to bottom-right.
function centers(n, side) {
  if (!n) return [];
  const sp = n === 1 ? 0 : Math.min(52, 150 / (n - 1));
  return Array.from({ length: n }, (_, i) => (side === "input" ? 180 + sp * ((n - 1) / 2 - i) : -sp * ((n - 1) / 2 - i)));
}
function eyes(x1, x2, cy) {
  return `<g class="eyes"><ellipse class="eye" cx="${x1}" cy="${cy}" rx="6" ry="10"/><ellipse class="eye" cx="${x2}" cy="${cy}" rx="6" ry="10"/></g>`;
}
const BODY = `<circle class="body" cx="100" cy="108" r="38"/>`;
function face(pose) {
  switch (pose) {
    case "listen": return BODY + eyes(83, 109, 97);
    case "think": return BODY + eyes(92, 118, 96);
    case "review": return BODY + eyes(95, 121, 104);
    case "block": return BODY + eyes(87, 113, 102) + `<polygon class="body" points="78,88 96,88 96,100 78,93"/><polygon class="body" points="122,88 104,88 104,100 122,93"/><path class="cut" d="M93 124 L107 124"/>`;
    case "reply": return BODY + eyes(87, 113, 100) + `<path class="cut" d="M91 120 Q100 128 109 120"/>`;
    case "idle": return BODY + `<path class="cut" d="M81 106 L93 106"/><path class="cut" d="M107 106 L119 106"/>`;
    default: return BODY + eyes(87, 113, 102);
  }
}
function bubble(points, rect) {
  const shape = `<polygon points="${points}"/><rect ${rect}/>`;
  return `<g class="b-out">${shape}</g><g class="b-in">${shape}</g>`;
}
function extras(pose, d, side) {
  if (pose === "listen") {
    return `<g transform="translate(${-d} ${-d * 0.6})"><g class="pop">${bubble("56,48 72,48 80,66", 'x="18" y="16" width="64" height="36" rx="14"')}<circle class="dot" cx="36" cy="34" r="4.5"/><circle class="dot d2" cx="50" cy="34" r="4.5"/><circle class="dot d3" cx="64" cy="34" r="4.5"/></g></g>`;
  }
  if (pose === "think") {
    return `<g transform="translate(${d} ${-d * 0.6})"><circle class="th" cx="158" cy="44" r="4"/><circle class="th t2" cx="171" cy="28" r="6"/><circle class="th t3" cx="187" cy="12" r="7"/></g>`;
  }
  if (pose === "block") {
    const mirror = side === "output" ? ' transform="translate(200 0) scale(-1 1)"' : "";
    const shape = `<polygon points="12,30 22,30 8,40"/><rect x="4" y="7" width="34" height="26" rx="14"/>`;
    return `<g${mirror}><g transform="translate(${-d} ${-d * 0.6})"><g class="pop"><g transform="rotate(-18 21 20)"><g class="d-out">${shape}</g><g class="d-in">${shape}</g><path class="d-x" d="M16 15 L26 25 M26 15 L16 25"/></g><g class="thin"><path d="M47 38 L42 32"/><path d="M54 35 L55 27"/><path d="M44 45 L37 46"/></g></g></g></g>`;
  }
  if (pose === "reply") {
    return `<g transform="translate(${d} ${d * 0.6})"><g class="pop">${bubble("160,170 172,166 162,154", 'x="152" y="166" width="44" height="30" rx="14"')}<path class="chk" d="M162 182 L169 189 L182 175"/></g></g>`;
  }
  if (pose === "idle") {
    return `<text class="zz" x="${146 + d}" y="${50 - d * 0.6}">z</text><text class="zz z2" x="${162 + d}" y="${32 - d * 0.6}">z</text>`;
  }
  return "";
}

const key = (side, name) => `${side}:${name}`;
const FINAL = {
  answered: "Answered",
  guided_refusal: "Guided refusal sent",
  orbital_response: "Answered by an orbital",
  blocked: "Blocked",
  fallback_refusal: "Fallback refusal sent",
};

export function createStage({ svgs, setCaption, onOrbital, onAtom, onFinished }) {
  const s = {
    pose: "rest", dash: {}, shield: null, blockSide: "input", busy: false, slow: true,
    inputs: [], outputs: [], kinds: {}, model: "", maxRetries: 2, lastInput: null, lastOutput: null,
    rest: null,
  };
  const queue = [];
  let draining = false;
  let idleTimer = 0;
  let restTimer = 0;
  const step = () => wait(s.slow ? 420 : 0);
  const hold = (ms) => wait(s.slow ? ms : 0);

  /** The ring of orbitals as it is drawn: its radius, and the scale of the whole drawing. */
  function ring() {
    const r = 66 + Math.max(0, Math.max(s.inputs.length, s.outputs.length) - 3) * 7;
    return { r, scale: 0.84 + 0.04 * Math.min(s.inputs.length + s.outputs.length, 4) };
  }

  function render() {
    const { r, scale } = ring();
    const d = (r - 66) * 0.75;
    let svg = "";
    const draw = (list, side) => {
      if (s.shield === side) {
        const [a0, a1] = side === "input" ? [125, 235] : [-55, 55];
        svg += `<path class="shield" d="${arc(r, a0, a1)}"/>`;
        return;
      }
      centers(list.length, side).forEach((c, i) => {
        const o = list[i];
        const state = s.dash[key(side, o.name)] || "";
        svg += `<path class="dash ${state}${s.pose === "idle" ? " dim" : ""}" data-orb="${esc(o.name)}" data-side="${side}" d="${arc(r, c - 8.35, c + 8.35)}"><title>${esc(o.name)}</title></path>`;
      });
    };
    draw(s.inputs, "input");
    draw(s.outputs, "output");
    svg += `<g class="nuc" data-atom="1"><title>nucleus</title>${face(s.pose)}</g>${extras(s.pose, d, s.blockSide)}`;
    for (const el of svgs) {
      const root = $(".atom-root", el);
      root.innerHTML = svg;
      root.style.transform = `scale(${scale})`;
      el.setAttribute("class", `atom pose-${s.pose}`);
    }
  }

  function restDashes() {
    s.dash = {};
    if (s.inputs.length) s.dash[key("input", s.inputs[0].name)] = "lit";
    if (s.outputs.length) s.dash[key("output", s.outputs[s.outputs.length - 1].name)] = "lit";
  }

  function toRest() {
    s.pose = "rest";
    s.shield = null;
    restDashes();
    const [title, sub] = s.rest || ["Waiting for a message", `${plural(s.inputs.length, "input orbital")} → nucleus → ${plural(s.outputs.length, "output orbital")}`];
    setCaption(title, sub);
    render();
    armIdle();
  }

  function armIdle() {
    clearTimeout(idleTimer);
    // Something is keeping the atom at work: it does not doze.
    if (s.rest) return;
    idleTimer = setTimeout(() => {
      if (!s.busy && s.pose === "rest") {
        s.pose = "idle";
        setCaption("Dozing", "Send a message to wake it up");
        render();
      }
    }, 45000);
  }

  function wake() {
    if (s.pose === "idle") toRest();
    else if (s.pose === "rest") armIdle();
  }
  document.addEventListener("pointerdown", wake);
  document.addEventListener("keydown", wake);

  svgs[0].addEventListener("click", (e) => {
    if (s.busy) return;
    const dash = e.target.closest("[data-orb]");
    if (dash) onOrbital(dash.dataset.side, dash.dataset.orb);
    else if (e.target.closest("[data-atom]")) onAtom();
  });

  function skipMissing(side) {
    for (const o of side === "input" ? s.inputs : s.outputs) {
      const k = key(side, o.name);
      if (!s.dash[k]) s.dash[k] = "skip";
    }
  }

  function finish(turn) {
    s.busy = false;
    skipMissing("input");
    skipMissing("output");
    const report = turn.report;
    const action = report ? report.final_action : "aborted";
    if (turn.error || action === "aborted") {
      s.pose = "block";
      s.shield = null;
      setCaption(`The turn failed: ${turn.error ? turn.error.type : report.error_type || "error"}`, turn.error ? turn.error.message : "", true);
    } else if (action === "answered" || action === "guided_refusal" || action === "orbital_response") {
      s.pose = "reply";
      s.shield = null;
      setCaption(FINAL[action], fmtMs(report.duration_ms));
    } else {
      s.pose = "block";
      s.shield = report.output.length ? "output" : "input";
      s.blockSide = s.shield;
      setCaption(FINAL[action] || "Blocked", fmtMs(report.duration_ms));
    }
    try {
      render();
    } finally {
      // Whatever happens to the drawing, the turn reaches the conversation.
      onFinished(turn);
      clearTimeout(restTimer);
      restTimer = setTimeout(() => { if (!s.busy) toRest(); }, 3800);
    }
  }

  async function apply(ev) {
    switch (ev.type) {
      case "turn_started":
        clearTimeout(restTimer);
        clearTimeout(idleTimer);
        Object.assign(s, { busy: true, dash: {}, shield: null, pose: "listen", lastInput: null, lastOutput: null });
        setCaption("Checking the message", `turn ${ev.n}`);
        render();
        break;
      case "orbital_started": {
        s.pose = ev.side === "input" ? "listen" : "review";
        s.shield = null;
        s.dash[key(ev.side, ev.name)] = "run";
        const kind = s.kinds[key(ev.side, ev.name)] || "orbital";
        setCaption(ev.side === "input" ? "Checking the message" : "Checking the reply",
          ev.side === "input" ? `${ev.name} · ${kind}` : `${ev.name} · ${kind} · attempt ${ev.attempt}`);
        render();
        await step();
        break;
      }
      case "orbital_finished":
        s.dash[key(ev.side, ev.name)] = ev.action === "block" ? "block" : "pass";
        if (ev.side === "input") s.lastInput = ev;
        else s.lastOutput = ev;
        render();
        await hold(160);
        break;
      case "input_decided": {
        skipMissing("input");
        const by = s.lastInput ? s.lastInput.name : "an orbital";
        if (ev.kind === "reply" && s.lastInput && s.lastInput.action === "respond") {
          s.pose = "reply";
          setCaption(`${by} answered`, "The nucleus is not called");
          render();
          await hold(700);
          break;
        }
        if (ev.kind !== "proceed") {
          Object.assign(s, { pose: "block", shield: "input", blockSide: "input" });
          setCaption(`Blocked by ${by}`, ev.kind === "guided_refusal"
            ? "The nucleus answers from its guidance, without seeing the message"
            : "The user gets the orbital's reply");
          render();
          await hold(1100);
        }
        if (ev.kind !== "reply") {
          Object.assign(s, { pose: "think", shield: null });
          setCaption(ev.kind === "guided_refusal" ? "Writing a guided refusal" : "The nucleus is writing", s.model);
          render();
        }
        break;
      }
      case "reply_drafted":
        for (const o of s.outputs) delete s.dash[key("output", o.name)];
        render();
        await hold(400);
        break;
      case "output_decided":
        if (ev.kind === "regenerate") {
          Object.assign(s, { pose: "block", shield: "output", blockSide: "output" });
          setCaption(`${s.lastOutput ? s.lastOutput.name : "An output orbital"} blocked the reply`,
            `Regenerating with its guidance · attempt ${ev.attempt + 1} of ${s.maxRetries + 1}`);
          render();
          await hold(1000);
          Object.assign(s, { pose: "think", shield: null });
          setCaption(`Regenerating · attempt ${ev.attempt + 1}`, s.model);
          render();
        }
        break;
      case "turn_finished":
        finish(ev.turn);
        break;
      default:
        break;
    }
  }

  async function drain() {
    draining = true;
    try {
      while (queue.length) {
        const ev = queue.shift();
        try {
          await apply(ev);
        } catch (error) {
          // A drawing failure must not stop the events that follow, the turn's end above all.
          console.error("chatomon studio: could not show", ev.type, error);
        }
      }
    } finally {
      draining = false;
    }
  }

  return {
    push(ev) { queue.push(ev); if (!draining) drain(); },
    setSlow(value) { s.slow = value; },
    ring,
    busy: () => s.busy,
    /** What the stage says while no turn runs, in place of its own words: `[title, sub]`, or
        `null` for its own. With words of another the atom stays awake. */
    setRest(caption) {
      s.rest = caption;
      if (!s.busy && (s.pose === "rest" || s.pose === "idle")) toRest();
    },
    /** A turn started before this page was loaded: its events are gone, show that it runs. */
    showRunning() {
      clearTimeout(restTimer);
      clearTimeout(idleTimer);
      Object.assign(s, { busy: true, pose: "think", shield: null, dash: {} });
      setCaption("A turn is running", "It started before this page was loaded");
      render();
    },
    release() {
      s.busy = false;
      toRest();
    },
    setOrbitals(orbitals, atom) {
      s.inputs = orbitals.input.filter((o) => o.enabled);
      s.outputs = orbitals.output.filter((o) => o.enabled);
      s.kinds = Object.fromEntries([...orbitals.input, ...orbitals.output].map((o) => [key(o.side, o.name), o.kind]));
      s.model = atom.model;
      s.maxRetries = atom.max_retries;
      if (!s.busy) toRest();
    },
  };
}

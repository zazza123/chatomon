import { $, esc, reduceMotion } from "../util.js";

/* The Spectral ghost: one drawing, six poses, and the ghost that lives on the stage. */

export const GHOST_POSES = ["sleep", "connect", "orbit", "evaluate", "lost", "refused"];

// The Pixel shape, in a box of its own whose centre is (50, 54): the body, and the square that
// came off its corner.
export const GHOST_BODY = "M14 46A36 36 0 0 1 86 46V98H72V82H57V98H43V82H26L14 70Z";
const NS = "http://www.w3.org/2000/svg";

function eyes(rx = 5.5, ry = 9, dy = 0) {
  return `<ellipse class="g-eye" cx="38.5" cy="${44 + dy}" rx="${rx}" ry="${ry}"/><ellipse class="g-eye" cx="61.5" cy="${44 + dy}" rx="${rx}" ry="${ry}"/>`;
}
function face(pose) {
  if (pose === "sleep") return `<path class="g-cut" d="M33 46h11M56 46h11"/>`;
  if (pose === "evaluate") return `<g class="g-eyes">${eyes()}</g><path class="g-cut" d="M43 60q7 7 14 0"/>`;
  if (pose === "refused") return `<g class="g-eyes">${eyes(6.5, 7.5, -2)}</g><ellipse class="g-eye" cx="50" cy="63" rx="4.5" ry="5.5"/>`;
  return `<g class="g-eyes">${eyes()}</g>`;
}

/** The ghost itself, in its own coordinates. */
export function ghostMarkup(pose) {
  return `<g class="g-all"><rect class="g-pix" x="0" y="90" width="14" height="14" rx="1.5"/><path class="g-body" d="${GHOST_BODY}"/><g class="g-look">${face(pose)}</g></g>`;
}

/** What a pose adds around the ghost, in the ghost's own coordinates. */
export function ghostExtras(pose) {
  if (pose === "sleep") return `<text class="g-zz" x="90" y="22">z</text><text class="g-zz g-z2" x="104" y="4">z</text>`;
  if (pose === "connect") return `<circle class="g-dot" cx="34" cy="120" r="5"/><circle class="g-dot g-d2" cx="50" cy="120" r="5"/><circle class="g-dot g-d3" cx="66" cy="120" r="5"/>`;
  if (pose === "refused") {
    const shape = `<polygon points="86,24 97,26 80,38"/><rect x="80" y="-16" width="44" height="34" rx="15"/>`;
    return `<g class="g-pop"><g class="g-no-out">${shape}</g><g class="g-no-in">${shape}</g><path class="g-no-x" d="M95 -6l14 14M109 -6l-14 14"/></g>`;
  }
  return "";
}

/** The ghost of the chip in the header: small, without what goes around it. */
export function ghostIcon(pose) {
  return `<svg class="ghost gicon pose-${pose}" viewBox="-8 2 116 112" aria-hidden="true">${ghostMarkup(pose)}</svg>`;
}

/** The ghost at the top of its card, where the card that is open hides the stage. */
export function ghostLive(pose, label) {
  const sending = pose === "evaluate"
    ? `<g class="g-pop"><rect class="gb-out" x="88" y="-12" width="34" height="24" rx="11"/><rect class="gb-fill" x="88" y="-12" width="34" height="24" rx="11"/><circle class="g-dot" cx="97" cy="0" r="3"/><circle class="g-dot g-d2" cx="105" cy="0" r="3"/><circle class="g-dot g-d3" cx="113" cy="0" r="3"/></g>`
    : "";
  return `<svg class="ghost glive pose-${pose}" viewBox="-8 -22 136 150" role="img" aria-label="${esc(label)}"><g class="g-bob">${ghostMarkup(pose)}</g>${ghostExtras(pose)}${sending}</svg>`;
}

/**
 * The ghost on the stage. Its corner is the bottom left, the one the atom's poses leave free.
 * Connected, it goes around the atom: behind it in the upper half of its orbit, in front of it
 * in the lower half. During a turn of the developer's it waits in its corner.
 */
export function createGhost({ svg, stage, onOpen }) {
  const make = (name, attrs = {}) => {
    const node = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    return node;
  };
  // Behind the atom, in front of it, and above everything for what flies between the two.
  const root = $(".atom-root", svg);
  const back = svg.insertBefore(make("g", { id: "gh-back" }), root);
  const front = svg.appendChild(make("g", { id: "gh-front" }));
  const fx = svg.appendChild(make("g", { id: "gh-fx" }));
  const squares = [0, 1, 2].map((i) => back.appendChild(make("rect", { class: `g-trail g-t${i}`, rx: 1, opacity: 0 })));
  const g = make("g", { id: "gh" });
  g.appendChild(make("title")).textContent = "Spectral";
  const posG = g.appendChild(make("g"));
  front.appendChild(g);
  g.addEventListener("click", onOpen);

  const COS = Math.cos((-16 * Math.PI) / 180);
  const SIN = Math.sin((-16 * Math.PI) / 180);
  // The orbit and the corner follow the size of the ring of orbitals, as far as the drawing goes:
  // with a large ring the corner stops at the edge, above the words under the stage.
  const geo = () => {
    const { r, scale } = stage.ring();
    const R = r * scale;
    return { rx: R + 54, ry: R + 8, hx: Math.max(100 - R - 42, 6), hy: Math.min(108 + R + 22, 194) };
  };
  const orbitPoint = (t) => {
    const { rx, ry } = geo();
    const ux = rx * Math.cos(t);
    const uy = ry * Math.sin(t);
    return { x: 100 + ux * COS - uy * SIN, y: 108 + ux * SIN + uy * COS };
  };

  let pose = "sleep";
  let shown = "";
  let theta = 0.35 * Math.PI;
  let last = 0;
  let moving = false;
  let look = null;
  let smileTimer = 0;
  const pos = { x: geo().hx, y: geo().hy, s: 0.52, f: 1 };
  const hist = [];
  const flying = [];
  const circling = () => (pose === "orbit" || pose === "evaluate") && !stage.busy();

  function place() {
    posG.setAttribute("transform", `translate(${pos.x.toFixed(2)} ${pos.y.toFixed(2)}) scale(${(pos.f * pos.s).toFixed(3)} ${pos.s.toFixed(3)}) translate(-50 -54)`);
  }
  function draw(face = pose) {
    if (shown === `${pose}:${face}`) return;
    shown = `${pose}:${face}`;
    g.setAttribute("class", `ghost pose-${pose}`);
    posG.innerHTML = ghostMarkup(face) + ghostExtras(pose);
    look = $(".g-look", posG);
    place();
  }

  function fly(kind) {
    const node = make("g");
    const mark = kind === "in"
      ? `<circle class="gb-dot" cx="-6" r="2.3"/><circle class="gb-dot" r="2.3"/><circle class="gb-dot" cx="6" r="2.3"/>`
      : `<path class="gb-chk" d="M-5.5 0l4 4.5l7.5 -8.5"/>`;
    node.innerHTML = `<rect class="gb-out" x="-13" y="-10" width="26" height="20" rx="9"/><rect class="gb-fill" x="-13" y="-10" width="26" height="20" rx="9"/>${mark}`;
    fx.appendChild(node);
    flying.push({ node, kind, t: 0, from: { x: pos.x, y: pos.y } });
  }
  function landed(item) {
    item.node.remove();
    if (item.kind === "in") {
      // The nucleus takes the message in, unless a turn of the developer's is on the stage.
      const nucleus = $(".nuc", svg);
      if (nucleus && !stage.busy()) {
        nucleus.classList.remove("gh-squash");
        nucleus.getBoundingClientRect();
        nucleus.classList.add("gh-squash");
      }
    } else if (pose === "orbit") {
      // The last reply of a run that has just ended: a smile, for a moment.
      draw("evaluate");
      clearTimeout(smileTimer);
      smileTimer = setTimeout(() => draw(), 550);
    }
  }

  function frame(now) {
    const dt = Math.min(48, now - (last || now)) / 1000;
    last = now;
    const { hx, hy } = geo();
    let target;
    let inFront = true;
    let faceTo = 1;
    if (circling()) {
      const before = theta;
      theta = (theta + (dt * 2 * Math.PI) / (pose === "evaluate" ? 3.6 : 9.5)) % (2 * Math.PI);
      const point = orbitPoint(theta);
      const ahead = orbitPoint(theta + 0.05);
      target = { x: point.x, y: point.y, s: 0.5 * (1 + 0.14 * Math.sin(theta)) };
      inFront = Math.sin(theta) > 0;
      faceTo = ahead.x >= point.x ? 1 : -1;
      if (pose === "evaluate") {
        // On the input side it hands a message over; on the output side it takes the reply.
        if (before < Math.PI && theta >= Math.PI) fly("in");
        if (theta < before) fly("out");
      }
    } else {
      // Asleep it lies still; awake it floats a little.
      target = { x: hx, y: hy + (pose === "sleep" ? 0 : Math.sin(now / 700) * 3), s: 0.52 };
    }
    const k = 1 - Math.exp(-dt * 7);
    pos.x += (target.x - pos.x) * k;
    pos.y += (target.y - pos.y) * k;
    pos.s += (target.s - pos.s) * k;
    pos.f += (faceTo - pos.f) * (1 - Math.exp(-dt * 12));
    if (Math.abs(pos.f) < 0.06) pos.f = pos.f < 0 ? -0.06 : 0.06;
    place();
    const layer = inFront ? front : back;
    if (g.parentNode !== layer) layer.appendChild(g);

    // Awake, it looks at the atom.
    const dx = Math.max(-1, Math.min(1, (100 - pos.x) / 110));
    const dy = Math.max(-1, Math.min(1, (104 - pos.y) / 110));
    if (look) look.setAttribute("transform", pose === "sleep" ? "" : `translate(${(dx * 3 * Math.sign(pos.f)).toFixed(2)} ${(dy * 2.5).toFixed(2)})`);

    // The trail: three squares where the ghost was a moment ago, while requests arrive.
    hist.push({ x: pos.x, y: pos.y });
    if (hist.length > 40) hist.shift();
    squares.forEach((rect, i) => {
      const past = hist[hist.length - 1 - (i + 1) * 9];
      const size = 9 - i * 2;
      if (pose !== "evaluate" || stage.busy() || !past) {
        rect.setAttribute("opacity", 0);
        return;
      }
      rect.setAttribute("x", (past.x - size / 2).toFixed(1));
      rect.setAttribute("y", (past.y - size / 2 + 4).toFixed(1));
      rect.setAttribute("width", size);
      rect.setAttribute("height", size);
      rect.setAttribute("opacity", 0.95 - i * 0.2);
    });

    for (let i = flying.length - 1; i >= 0; i -= 1) {
      const item = flying[i];
      item.t += dt / 0.5;
      const p = Math.min(1, item.t);
      const ease = p * p * (3 - 2 * p);
      const a = item.kind === "in" ? item.from : { x: 100, y: 108 };
      const b = item.kind === "in" ? { x: 100, y: 108 } : { x: pos.x, y: pos.y - 6 };
      const size = item.kind === "in" ? 1 - 0.6 * ease : 0.4 + 0.6 * ease;
      item.node.setAttribute("transform", `translate(${(a.x + (b.x - a.x) * ease).toFixed(1)} ${(a.y + (b.y - a.y) * ease - Math.sin(p * Math.PI) * 14).toFixed(1)}) scale(${size.toFixed(2)})`);
      if (p >= 1) {
        flying.splice(i, 1);
        landed(item);
      }
    }
    // Asleep in its corner with nothing in the air, the drawing no longer changes: no frame is
    // asked for until something does.
    const rested = Math.abs(pos.x - target.x) < 0.05 && Math.abs(pos.y - target.y) < 0.05 && Math.abs(pos.s - target.s) < 0.002 && pos.f > 0.99;
    if (pose === "sleep" && rested && !flying.length) {
      moving = false;
      return;
    }
    requestAnimationFrame(frame);
  }
  function move() {
    if (moving) return;
    moving = true;
    last = 0;
    requestAnimationFrame(frame);
  }

  /** Without motion every pose is a still drawing: at home, or at a fixed point of its orbit. */
  function still() {
    const point = circling() ? orbitPoint(0.35 * Math.PI) : { x: geo().hx, y: geo().hy };
    Object.assign(pos, { x: point.x, y: point.y, s: 0.52, f: circling() ? -1 : 1 });
    place();
    if (g.parentNode !== front) front.appendChild(g);
  }

  draw();
  if (reduceMotion) still();
  else move();

  return {
    /** Show another pose. The ghost goes where the pose belongs by itself. */
    set(next) {
      pose = next;
      clearTimeout(smileTimer);
      draw();
      if (reduceMotion) still();
      else move();
    },
    /** The ring changed its size, or a turn started or ended: a ghost at rest goes to its place. */
    settle() {
      if (reduceMotion) still();
      else move();
    },
  };
}

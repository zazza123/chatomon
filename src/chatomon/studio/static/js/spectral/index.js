import { $ } from "../util.js";
import { createCard } from "./card.js";
import { createGhost } from "./ghost.js";

/* The Spectral add-on of the page. `main.js` asks for this file only when the state of the
   studio has a `spectral` entry: without the add-on's extra nothing of it is requested. */

function styles() {
  return new Promise((resolve) => {
    const sheet = document.createElement("link");
    sheet.rel = "stylesheet";
    sheet.href = "/static/spectral.css";
    // Shown with or without its styles: a page that waits for a file forever shows nothing.
    sheet.onload = resolve;
    sheet.onerror = resolve;
    document.head.append(sheet);
  });
}

/** Put the add-on in the page: its chip in the header, its ghost on the stage, its card. */
export async function mount({ stage, toast }) {
  await styles();

  const chip = document.createElement("button");
  chip.type = "button";
  chip.id = "spectral-chip";
  chip.className = "chip-spectral";
  const live = document.createElement("span");
  live.id = "spectral-live";
  live.className = "sr";
  live.setAttribute("role", "status");
  $("#content-chip").before(chip, live);

  $(".legend span:last-child").textContent = "Click a dash, the nucleus or the ghost to open it";

  let card = null;
  const ghost = createGhost({ svg: $("#atom"), stage, onOpen: () => card.show() });
  card = createCard({ chip, live, ghost, stage, toast });
  return { sync: card.sync };
}

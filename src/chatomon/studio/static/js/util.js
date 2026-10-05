export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
export const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
/** A sentence from the server: escaped, with what it puts between backticks shown as code. */
export const rich = (text) => esc(text).replace(/`([^`]+)`/g, "<code>$1</code>");
export const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
export const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
export const fmtMs = (ms) => (ms < 10 ? `${ms.toFixed(1)} ms` : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`);
export const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
export const shortFp = (fp) => (fp && fp.length > 20 ? `${fp.slice(0, 19)}…` : fp ?? "");
export const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* storage unavailable */ } },
};

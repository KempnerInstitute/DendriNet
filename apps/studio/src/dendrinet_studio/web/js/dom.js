// Small DOM helpers. All text goes in through textContent; nothing is parsed as HTML.

export const SVG_NS = "http://www.w3.org/2000/svg";

export function h(tag, attrs = {}, children = []) {
  const element = document.createElement(tag);
  setAttributes(element, attrs);
  append(element, children);
  return element;
}

export function s(tag, attrs = {}, children = []) {
  const element = document.createElementNS(SVG_NS, tag);
  setAttributes(element, attrs);
  append(element, children);
  return element;
}

function setAttributes(element, attrs) {
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") element.setAttribute("class", value);
    else if (key === "text") element.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") element.addEventListener(key.slice(2), value);
    else if (key === "dataset") Object.assign(element.dataset, value);
    else if (value === true) element.setAttribute(key, "");
    else element.setAttribute(key, String(value));
  }
}

function append(element, children) {
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function clear(element) {
  element.replaceChildren();
  return element;
}

export const $ = id => document.getElementById(id);

export function fmt(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (typeof value !== "number") return String(value);
  if (digits !== undefined) return value.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
  return Number.isInteger(value) ? value.toLocaleString("en-US") : value.toLocaleString("en-US", { maximumSignificantDigits: 4 });
}

export function pct(value) {
  return value === null || value === undefined ? "—" : (100 * value).toFixed(1) + "%";
}

export function duration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return Math.round(seconds) + " s";
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m} min ${Math.round(seconds - 60 * m)} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

export function timeOfDay(epochSeconds) {
  if (!epochSeconds) return "";
  const d = new Date(epochSeconds * 1000);
  const today = new Date();
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return d.toDateString() === today.toDateString() ? time : `${d.toLocaleDateString()} ${time}`;
}

// Icons: 16 px line icons drawn with currentColor.
const ICONS = {
  check: "M3 8.5l3 3 7-7",
  cross: "M4 4l8 8M12 4l-8 8",
  warn: "M8 2l6.5 11.5h-13zM8 6.5v3.5M8 12v.5",
  dot: "M8 8m-3 0a3 3 0 1 0 6 0a3 3 0 1 0 -6 0",
  play: "M5 3.5v9l7-4.5z",
  stop: "M4.5 4.5h7v7h-7z",
  undo: "M5 6H10a3 3 0 0 1 0 6H7M5 6l2.5-2.5M5 6l2.5 2.5",
  redo: "M11 6H6a3 3 0 0 0 0 6h3M11 6L8.5 3.5M11 6L8.5 8.5",
  plus: "M8 3v10M3 8h10",
  minus: "M3 8h10",
  download: "M8 2.5v8M4.5 7L8 10.5 11.5 7M3 13.5h10",
  upload: "M8 10.5v-8M4.5 6L8 2.5 11.5 6M3 13.5h10",
  copy: "M5.5 5.5h7v7h-7zM3.5 10.5v-7h7",
  fit: "M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10",
  zoomIn: "M7 3.5a3.5 3.5 0 1 0 0 7a3.5 3.5 0 1 0 0-7M12.5 12.5l-3-3M7 5.5v3M5.5 7h3",
  zoomOut: "M7 3.5a3.5 3.5 0 1 0 0 7a3.5 3.5 0 1 0 0-7M12.5 12.5l-3-3M5.5 7h3",
  help: "M8 14a6 6 0 1 0 0-12a6 6 0 1 0 0 12M6.2 6.2a1.9 1.9 0 1 1 2.6 1.8c-.6.3-.8.7-.8 1.3M8 11.3v.2",
  trash: "M3.5 4.5h9M6.5 4.5V3h3v1.5M5 4.5l.5 8.5h5l.5-8.5",
  folder: "M2.5 4.5h4l1 1.5h6v6.5h-11z",
  search: "M7 3a4 4 0 1 0 0 8a4 4 0 1 0 0-8M13 13l-3.2-3.2",
  chevron: "M6 4l4 4-4 4",
  layers: "M8 2.5l5.5 3-5.5 3-5.5-3zM2.5 8.5l5.5 3 5.5-3",
  sliders: "M3 4.5h10M3 11.5h10M6 3v3M10 10v3",
  flask: "M6.5 2.5h3M7 2.5v4L3.5 12.5a1 1 0 0 0 .9 1.5h7.2a1 1 0 0 0 .9-1.5L9 6.5v-4",
  gear: "M8 5.5a2.5 2.5 0 1 0 0 5a2.5 2.5 0 1 0 0-5M8 1.5v2M8 12.5v2M1.5 8h2M12.5 8h2M3.4 3.4l1.4 1.4M11.2 11.2l1.4 1.4M3.4 12.6l1.4-1.4M11.2 4.8l1.4-1.4",
  network: "M3.5 4a1.5 1.5 0 1 0 0 .01M3.5 12a1.5 1.5 0 1 0 0 .01M12.5 8a1.5 1.5 0 1 0 0 .01M5 4.3l6 3.2M5 11.7l6-3.2"
};

export function icon(name, label) {
  const svg = s("svg", { class: "icon", viewBox: "0 0 16 16", width: 16, height: 16, "aria-hidden": label ? undefined : "true", role: label ? "img" : undefined, "aria-label": label });
  svg.append(s("path", { d: ICONS[name] || ICONS.dot, fill: "none", stroke: "currentColor", "stroke-width": 1.5, "stroke-linecap": "round", "stroke-linejoin": "round" }));
  return svg;
}

// Glyphs shared by the outline, the canvas and the inspector: a triangle for
// excitatory (pyramidal) somata and a circle for inhibitory ones, as in the papers.
export function polarityGlyph(polarity, size = 14) {
  const svg = s("svg", { class: "glyph " + (polarity === "inhibitory" ? "inh" : "exc"), width: size, height: size, viewBox: "0 0 16 16", "aria-hidden": "true" });
  if (polarity === "inhibitory") svg.append(s("circle", { cx: 8, cy: 8.5, r: 5.2 }));
  else svg.append(s("path", { d: "M8 2.5L14 13.5H2Z" }));
  return svg;
}

export function debounce(fn, ms) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

export function download(filename, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h("a", { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    return false;
  }
}

// Run comparison charts. A run keeps its colour slot and marker shape while it
// stays selected, so identity never depends on colour alone: every run also has
// its own marker and a text label at the end of its curve, and the run table
// lists every value shown here.

import { h, s, fmt, pct } from "./dom.js";

export const SLOTS = 8;
const MARKERS = ["circle", "square", "triangle", "diamond", "triangleDown", "plus", "cross", "star"];

export function markerPath(kind, x, y, r = 4) {
  switch (kind) {
    case "square":
      return `M${x - r} ${y - r}h${2 * r}v${2 * r}h${-2 * r}z`;
    case "triangle":
      return `M${x} ${y - r * 1.15}L${x + r * 1.1} ${y + r * 0.85}H${x - r * 1.1}z`;
    case "triangleDown":
      return `M${x} ${y + r * 1.15}L${x + r * 1.1} ${y - r * 0.85}H${x - r * 1.1}z`;
    case "diamond":
      return `M${x} ${y - r * 1.25}L${x + r * 1.1} ${y}L${x} ${y + r * 1.25}L${x - r * 1.1} ${y}z`;
    case "plus":
      return `M${x - r} ${y - 1.3}h${r - 1.3}v${-(r - 1.3)}h2.6v${r - 1.3}h${r - 1.3}v2.6h${-(r - 1.3)}v${r - 1.3}h-2.6v${-(r - 1.3)}h${-(r - 1.3)}z`;
    case "cross": {
      const a = r * 0.85;
      return `M${x - a} ${y - a + 1.6}l1.6 -1.6l${a - 1.6} ${a - 1.6}l${a - 1.6} ${-(a - 1.6)}l1.6 1.6l${-(a - 1.6)} ${a - 1.6}l${a - 1.6} ${a - 1.6}l-1.6 1.6l${-(a - 1.6)} ${-(a - 1.6)}l${-(a - 1.6)} ${a - 1.6}l-1.6 -1.6l${a - 1.6} ${-(a - 1.6)}z`;
    }
    case "star": {
      let d = "";
      for (let i = 0; i < 10; i++) {
        const radius = i % 2 ? r * 0.5 : r * 1.25;
        const angle = -Math.PI / 2 + (i * Math.PI) / 5;
        d += `${i ? "L" : "M"}${x + radius * Math.cos(angle)} ${y + radius * Math.sin(angle)}`;
      }
      return d + "z";
    }
    default:
      return `M${x - r} ${y}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`;
  }
}

export const markerFor = slot => MARKERS[slot % MARKERS.length];

// A swatch that identifies a run everywhere: its colour and its marker.
export function runSwatch(slot, { open = false, size = 14 } = {}) {
  const svg = s("svg", { class: "swatch", width: size, height: size, viewBox: "0 0 14 14", "aria-hidden": "true" });
  svg.append(s("path", { d: markerPath(markerFor(slot), 7, 7, 4.2), class: `mk slot-${slot}${open ? " open" : ""}` }));
  return svg;
}

function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) {
    const pad = Math.abs(lo) * 0.1 || 1;
    lo -= pad;
    hi += pad;
  }
  const span = hi - lo;
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * power).find(m => span / m <= count) || 10 * power;
  const start = Math.floor(lo / step) * step;
  const end = Math.ceil(hi / step) * step;
  const ticks = [];
  for (let v = start; v <= end + step / 2; v += step) ticks.push(Number(v.toPrecision(12)));
  return { ticks, lo: start, hi: end, step };
}

function tickText(v, step) {
  const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return v.toFixed(Math.min(decimals, 6));
}

// Spread end labels so they never overlap.
function spread(items, minGap, top, bottom) {
  items.sort((a, b) => a.y - b.y);
  for (let i = 1; i < items.length; i++) items[i].y = Math.max(items[i].y, items[i - 1].y + minGap);
  const overflow = items.length ? items.at(-1).y - bottom : 0;
  if (overflow > 0) {
    items.at(-1).y -= overflow;
    for (let i = items.length - 2; i >= 0; i--) items[i].y = Math.min(items[i].y, items[i + 1].y - minGap);
  }
  for (const item of items) item.y = Math.max(top, item.y);
  return items;
}

// Loss (or any per-epoch measure) for several runs on shared axes.
// runs: [{id, label, slot, train: number[], valid: number[]}]
export function lineChart(container, runs, { yLabel = "Loss", height = 300 } = {}) {
  container.replaceChildren();
  const series = [];
  for (const run of runs) {
    if (run.train?.length) series.push({ run, split: "train", values: run.train });
    if (run.valid?.length) series.push({ run, split: "valid", values: run.valid });
  }
  if (!series.length) {
    container.append(h("p", { class: "empty", text: "No per-epoch values yet. Curves appear when a training run writes its summary." }));
    return;
  }
  const width = Math.max(420, container.clientWidth || 640);
  const labelRoom = Math.min(220, Math.max(130, width * 0.24));
  const m = { l: 56, r: labelRoom, t: 14, b: 42 };
  const W = width - m.l - m.r;
  const H = height - m.t - m.b;
  const finite = series.flatMap(sr => sr.values.filter(Number.isFinite));
  const yTicks = niceTicks(Math.min(...finite), Math.max(...finite));
  const epochs = Math.max(...series.map(sr => sr.values.length));
  const xTicks = niceTicks(1, Math.max(2, epochs), Math.min(8, epochs));
  const x = e => m.l + ((e - 1) / Math.max(1, epochs - 1)) * W;
  const y = v => m.t + H - ((v - yTicks.lo) / (yTicks.hi - yTicks.lo)) * H;
  const svg = s("svg", { class: "chart", width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${yLabel} by epoch for ${runs.length} run${runs.length === 1 ? "" : "s"}`, tabindex: 0 });
  for (const t of yTicks.ticks) {
    svg.append(s("line", { x1: m.l, x2: m.l + W, y1: y(t), y2: y(t), class: "grid" }));
    svg.append(s("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end", class: "tick" }, [tickText(t, yTicks.step)]));
  }
  for (const t of xTicks.ticks.filter(t => t >= 1 && t <= epochs && Number.isInteger(t))) {
    svg.append(s("line", { x1: x(t), x2: x(t), y1: m.t + H, y2: m.t + H + 4, class: "axis" }));
    svg.append(s("text", { x: x(t), y: m.t + H + 18, "text-anchor": "middle", class: "tick" }, [String(t)]));
  }
  svg.append(s("line", { x1: m.l, x2: m.l + W, y1: m.t + H, y2: m.t + H, class: "axis" }));
  svg.append(s("text", { x: m.l + W / 2, y: height - 6, "text-anchor": "middle", class: "axis-label" }, ["Epoch"]));
  svg.append(s("text", { x: 14, y: m.t + H / 2, transform: `rotate(-90 14 ${m.t + H / 2})`, "text-anchor": "middle", class: "axis-label" }, [yLabel]));

  const every = Math.max(1, Math.ceil(epochs / 6));
  const ends = [];
  for (const sr of series) {
    const points = sr.values.map((v, i) => [i + 1, v]).filter(([, v]) => Number.isFinite(v));
    if (!points.length) continue;
    const d = points.map(([e, v], i) => `${i ? "L" : "M"}${x(e).toFixed(1)} ${y(v).toFixed(1)}`).join("");
    svg.append(s("path", { d, class: `line slot-${sr.run.slot}${sr.split === "valid" ? " dashed" : ""}` }));
    const kind = markerFor(sr.run.slot);
    for (const [e, v] of points) {
      if ((e - 1) % every === 0 || e === points.at(-1)[0]) {
        svg.append(s("path", { d: markerPath(kind, x(e), y(v), 4), class: `mk slot-${sr.run.slot}${sr.split === "valid" ? " open" : ""}` }));
      }
    }
    const [le, lv] = points.at(-1);
    ends.push({ y: y(lv), x: x(le), text: `${sr.run.label} · ${sr.split === "valid" ? "val" : "train"}`, sr });
  }
  for (const item of spread(ends, 14, m.t + 4, m.t + H)) {
    const g = s("g", { class: "end-label" });
    g.append(s("path", { d: markerPath(markerFor(item.sr.run.slot), item.x + 14, item.y, 3.6), class: `mk slot-${item.sr.run.slot}${item.sr.split === "valid" ? " open" : ""}` }));
    const maxChars = Math.floor((labelRoom - 26) / 6.3);
    const text = item.text.length > maxChars ? item.text.slice(0, maxChars - 1) + "…" : item.text;
    g.append(s("text", { x: item.x + 22, y: item.y + 4, class: "end-text" }, [text]));
    if (text !== item.text) g.append(s("title", {}, [item.text]));
    svg.append(g);
  }

  // Crosshair and tooltip: the values of every curve at the nearest epoch.
  const guide = s("line", { y1: m.t, y2: m.t + H, class: "crosshair", visibility: "hidden" });
  svg.append(guide);
  const tip = h("div", { class: "chart-tip", hidden: true, role: "status" });
  const overlay = s("rect", { x: m.l, y: m.t, width: W, height: H, class: "hover-target" });
  svg.append(overlay);
  let focusEpoch = null;
  const show = epoch => {
    focusEpoch = Math.min(epochs, Math.max(1, epoch));
    guide.setAttribute("x1", x(focusEpoch));
    guide.setAttribute("x2", x(focusEpoch));
    guide.setAttribute("visibility", "visible");
    tip.replaceChildren(h("div", { class: "tip-head", text: `Epoch ${focusEpoch}` }));
    const rows = series
      .map(sr => ({ sr, v: sr.values[focusEpoch - 1] }))
      .filter(r => Number.isFinite(r.v))
      .sort((a, b) => a.v - b.v);
    for (const { sr, v } of rows) {
      tip.append(h("div", { class: "tip-row" }, [runSwatch(sr.run.slot, { open: sr.split === "valid", size: 12 }), h("span", { text: `${sr.run.label} · ${sr.split === "valid" ? "validation" : "training"}` }), h("b", { text: fmt(v, 4) })]));
    }
    tip.hidden = false;
    const left = x(focusEpoch) + 14;
    tip.style.left = `${Math.min(left, width - 260)}px`;
    tip.style.top = `${m.t + 6}px`;
  };
  const hide = () => {
    guide.setAttribute("visibility", "hidden");
    tip.hidden = true;
  };
  overlay.addEventListener("mousemove", event => {
    const box = svg.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * width;
    show(Math.round(1 + ((px - m.l) / W) * (epochs - 1)));
  });
  overlay.addEventListener("mouseleave", hide);
  svg.addEventListener("keydown", event => {
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      event.preventDefault();
      show((focusEpoch ?? (event.key === "ArrowRight" ? 0 : epochs + 1)) + (event.key === "ArrowRight" ? 1 : -1));
    } else if (event.key === "Escape") hide();
  });
  svg.addEventListener("blur", hide);
  container.append(h("div", { class: "chart-wrap" }, [svg, tip]));
}

// Final values of one metric: one row per run, a marker shape per split.
// rows: [{id, label, sub, slot, values: {train, valid, test}}]
const SPLITS = [
  ["train", "training", "circle", false, -7],
  ["valid", "validation", "diamond", true, 0],
  ["test", "test", "triangle", false, 7],
];

export function dotChart(container, rows, { metric = "accuracy", fraction = true, chance = null } = {}) {
  container.replaceChildren();
  const values = rows.flatMap(r => SPLITS.map(([k]) => r.values[k]).filter(Number.isFinite));
  if (!values.length) {
    container.append(h("p", { class: "empty", text: `No ${metric.replace(/_/g, " ")} recorded for the selected runs.` }));
    return;
  }
  const width = Math.max(360, container.clientWidth || 480);
  const rowH = 46;
  const m = { l: Math.min(200, width * 0.36), r: 22, t: 14, b: 42 };
  const height = m.t + m.b + rows.length * rowH;
  const W = width - m.l - m.r;
  let lo = 0;
  let hi = 1;
  if (!fraction) {
    const t = niceTicks(Math.min(...values, chance ?? Infinity), Math.max(...values, chance ?? -Infinity));
    lo = t.lo;
    hi = t.hi;
  }
  const ticks = fraction ? { ticks: [0, 0.25, 0.5, 0.75, 1], step: 0.25 } : niceTicks(lo, hi);
  const x = v => m.l + ((v - lo) / (hi - lo)) * W;
  const svg = s("svg", { class: "chart", width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `Final ${metric} of ${rows.length} runs` });
  for (const t of ticks.ticks) {
    svg.append(s("line", { x1: x(t), x2: x(t), y1: m.t, y2: height - m.b, class: "grid" }));
    svg.append(s("text", { x: x(t), y: height - m.b + 18, "text-anchor": "middle", class: "tick" }, [fraction ? `${Math.round(t * 100)}%` : tickText(t, ticks.step)]));
  }
  svg.append(s("text", { x: m.l + W / 2, y: height - 6, "text-anchor": "middle", class: "axis-label" }, [humanMetric(metric)]));
  if (Number.isFinite(chance)) {
    svg.append(s("line", { x1: x(chance), x2: x(chance), y1: m.t, y2: height - m.b, class: "chance" }));
    svg.append(s("text", { x: x(chance) + 4, y: m.t + 10, class: "tick" }, ["chance"]));
  }
  rows.forEach((row, i) => {
    const cy = m.t + rowH * i + rowH / 2;
    svg.append(s("line", { x1: m.l, x2: m.l + W, y1: cy, y2: cy, class: "row-rule" }));
    const label = row.label.length > 28 ? row.label.slice(0, 27) + "…" : row.label;
    svg.append(s("text", { x: m.l - 12, y: cy - 2, "text-anchor": "end", class: "row-label" }, [label]));
    if (row.sub) svg.append(s("text", { x: m.l - 12, y: cy + 13, "text-anchor": "end", class: "row-sub" }, [row.sub]));
    for (const [key, name, shape, open, dy] of SPLITS) {
      const v = row.values[key];
      if (!Number.isFinite(v)) continue;
      const mark = s("path", { d: markerPath(shape, x(v), cy + dy, 4.6), class: `mk slot-${row.slot}${open ? " open" : ""}`, tabindex: 0 });
      mark.append(s("title", {}, [`${row.label} · ${name}: ${fraction ? pct(v) : fmt(v, 4)}`]));
      svg.append(mark);
    }
  });
  const legend = h("div", { class: "chart-legend" }, SPLITS.map(([, name, shape, open]) => {
    const sw = s("svg", { width: 14, height: 14, viewBox: "0 0 14 14", "aria-hidden": "true" }, [s("path", { d: markerPath(shape, 7, 7, 4.4), class: `mk neutral${open ? " open" : ""}` })]);
    return h("span", { class: "legend-item" }, [sw, name]);
  }));
  if (Number.isFinite(chance)) legend.append(h("span", { class: "legend-item" }, [s("svg", { width: 18, height: 14, viewBox: "0 0 18 14", "aria-hidden": "true" }, [s("path", { d: "M2 7H16", class: "chance" })]), "chance"]));
  container.append(legend, svg);
}

export function humanMetric(name) {
  const known = { accuracy: "Accuracy", auc: "AUC", categorical_loglikelihood: "Log-likelihood per sample", mse: "Mean squared error", r2: "R²" };
  return known[name] || name.replace(/_/g, " ").replace(/^./, c => c.toUpperCase());
}

export function isFractionMetric(name) {
  return ["accuracy", "auc", "balanced_accuracy", "top5_accuracy", "f1"].includes(name);
}

export function curvesCsv(runs) {
  const epochs = Math.max(0, ...runs.flatMap(r => [r.train?.length || 0, r.valid?.length || 0]));
  const header = ["epoch", ...runs.flatMap(r => [`${r.label} train`, `${r.label} validation`])];
  const quote = v => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
  const lines = [header.map(quote).join(",")];
  for (let e = 0; e < epochs; e++) {
    lines.push([e + 1, ...runs.flatMap(r => [r.train?.[e] ?? "", r.valid?.[e] ?? ""])].join(","));
  }
  return lines.join("\n") + "\n";
}

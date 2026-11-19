// The neuron drawing, in the visual language of the DendriNet papers: hairline
// dendrites, a filled soma (triangle for excitatory, circle for inhibitory
// cells), excitatory contacts as round boutons and inhibitory contacts as bars.
// The tree is the exact branching-factor tree; large trees show a few branches
// per node with the true count written next to them, never a blank panel.

import { s, h, fmt } from "./dom.js";

const W = 640;
const H = 300;
const TOP = 30;
const BOTTOM = H - 40;
const MAX_SHOWN = 6;
const LEVEL_NAMES = (depth, deepest) => (depth === 0 ? "soma" : depth === deepest ? `${depth} · distal` : depth === 1 ? "1 · primary" : String(depth));

// counts: {ff_excitatory: [soma, d1, ...], ff_inhibitory: [...], rec_...}
export function drawNeuron(svg, { polarity = "excitatory", factors = [], counts = {}, highlight = null, title = "" }) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const depthCount = factors.length;
  const tree = buildTree(factors, 0);
  layout(tree, 70, W - 30);
  const yOf = depth => (depthCount === 0 ? BOTTOM - 40 : BOTTOM - 14 - (depth * (BOTTOM - 14 - TOP)) / depthCount);
  const somaX = tree.x;
  const somaY = BOTTOM;
  const segments = [];
  const ellipses = [];
  walk(tree, node => {
    node.y = node.depth === 0 ? somaY : yOf(node.depth);
    for (const child of node.children) {
      segments.push([node, child]);
    }
    if (node.hidden > 0) ellipses.push(node);
  });
  // Level guides and names.
  for (let depth = 0; depth <= depthCount; depth++) {
    const y = depth === 0 ? somaY : yOf(depth);
    const active = highlight && highlight.includes(depth);
    svg.append(s("text", { x: 8, y: y + 4, class: "level-name" + (active ? " active" : "") }, [LEVEL_NAMES(depth, depthCount)]));
    if (depth > 0) svg.append(s("line", { x1: 62, x2: W - 10, y1: y, y2: y, class: "level-guide" + (active ? " active" : "") }));
  }
  // Dendrites.
  for (const [parent, child] of segments) {
    const startY = parent.depth === 0 ? somaY - 12 : parent.y;
    svg.append(s("path", { d: `M${parent.x} ${startY}L${child.x} ${child.y}`, class: "dendrite" }));
  }
  for (const node of ellipses) {
    const kids = node.children;
    const last = kids.at(-1);
    svg.append(s("text", { x: last.x + 10, y: last.y + 4, class: "more" }, [`×${node.factor}`]));
  }
  // Contacts on each drawn compartment, drawn only while they stay legible:
  // where neighbouring branches of a level are close, the level is labelled once.
  const perComp = depth => ["ff_excitatory", "ff_inhibitory", "rec_excitatory", "rec_inhibitory"].map(k => (counts[k] || [])[depth] || 0);
  const xsByDepth = new Map();
  walk(tree, node => {
    if (node.depth > 0) xsByDepth.set(node.depth, [...(xsByDepth.get(node.depth) || []), node.x]);
  });
  const crowded = new Set();
  for (const [depth, xs] of xsByDepth) {
    xs.sort((a, b) => a - b);
    for (let i = 1; i < xs.length; i++) if (xs[i] - xs[i - 1] < 24) crowded.add(depth);
  }
  for (const depth of crowded) {
    const [fe, fi, re, ri] = perComp(depth);
    if (fe + fi + re + ri) svg.append(s("text", { x: W - 12, y: yOf(depth) - 5, "text-anchor": "end", class: "count-label" }, [contactText(fe, fi, re, ri)]));
  }
  for (const [parent, child] of segments) {
    const [fe, fi, re, ri] = perComp(child.depth);
    const total = fe + fi + re + ri;
    if (!total || crowded.has(child.depth)) continue;
    const startY = parent.depth === 0 ? somaY - 12 : parent.y;
    const dx = child.x - parent.x;
    const dy = child.y - startY;
    const length = Math.hypot(dx, dy);
    const room = length / (total + 1);
    if (room < 5 || total > 24) {
      const mx = parent.x + dx * 0.55;
      const my = startY + dy * 0.55;
      svg.append(s("text", { x: mx + 6, y: my, class: "count-label" }, [contactText(fe, fi, re, ri)]));
      continue;
    }
    const ux = dx / length;
    const uy = dy / length;
    const order = interleave([["fe", fe], ["fi", fi], ["re", re], ["ri", ri]]);
    order.forEach((kind, j) => {
      const t = 0.12 + (0.8 * (j + 0.5)) / order.length;
      const side = j % 2 ? 1 : -1;
      contact(svg, parent.x + dx * t, startY + dy * t, -uy * side, ux * side, ux, uy, kind);
    });
  }
  // Somatic contacts on the two flanks of the soma.
  const [se, si, sre, sri] = perComp(0);
  const somaKinds = interleave([["fe", se], ["fi", si], ["re", sre], ["ri", sri]]);
  if (somaKinds.length && somaKinds.length <= 24) {
    const half = Math.ceil(somaKinds.length / 2);
    somaKinds.forEach((kind, j) => {
      const left = j < half;
      const k = left ? j : j - half;
      const n = left ? half : somaKinds.length - half;
      const angle = left ? Math.PI * (0.62 + (0.56 * (k + 0.5)) / n) : Math.PI * (1.82 + (0.56 * (k + 0.5)) / n);
      const cx = somaX + Math.cos(angle) * 17;
      const cy = somaY - 2 + Math.sin(angle) * 17;
      contact(svg, cx, cy, Math.cos(angle), Math.sin(angle), -Math.sin(angle), Math.cos(angle), kind);
    });
  } else if (somaKinds.length) {
    svg.append(s("text", { x: somaX + 22, y: somaY + 4, class: "count-label" }, [contactText(se, si, sre, sri)]));
  }
  // Soma last, on top.
  if (polarity === "inhibitory") svg.append(s("circle", { cx: somaX, cy: somaY - 2, r: 10, class: "soma inh" }));
  else svg.append(s("path", { d: `M${somaX - 12} ${somaY + 9}L${somaX} ${somaY - 13}L${somaX + 12} ${somaY + 9}Z`, class: "soma exc" }));
  svg.setAttribute("aria-label", title || "Neuron morphology");
}

function contactText(fe, fi, re, ri) {
  const parts = [];
  if (fe) parts.push(`${fe} E`);
  if (fi) parts.push(`${fi} I`);
  if (re) parts.push(`${re} rec E`);
  if (ri) parts.push(`${ri} rec I`);
  return parts.join(" · ") + " each";
}

// Spread inhibitory contacts among the excitatory ones.
function interleave(groups) {
  const pools = groups.filter(([, n]) => n > 0).map(([k, n]) => ({ k, n }));
  const total = pools.reduce((a, p) => a + p.n, 0);
  const out = [];
  const taken = Object.fromEntries(pools.map(p => [p.k, 0]));
  for (let i = 0; i < total; i++) {
    let best = null;
    let bestLag = -Infinity;
    for (const p of pools) {
      if (taken[p.k] >= p.n) continue;
      const lag = (i + 1) * (p.n / total) - taken[p.k];
      if (lag > bestLag) {
        bestLag = lag;
        best = p;
      }
    }
    taken[best.k] += 1;
    out.push(best.k);
  }
  return out;
}

// One contact: a stalk perpendicular to the dendrite, then a bouton or a bar.
function contact(svg, x, y, nx, ny, ux, uy, kind) {
  const ex = x + nx * 6.5;
  const ey = y + ny * 6.5;
  const recurrent = kind === "re" || kind === "ri";
  if (kind === "fe" || kind === "re") {
    svg.append(s("path", { d: `M${x} ${y}L${ex} ${ey}`, class: "stalk exc" }));
    svg.append(s("circle", { cx: ex + nx * 1.8, cy: ey + ny * 1.8, r: 2.4, class: "bouton" + (recurrent ? " rec" : "") }));
  } else {
    svg.append(s("path", { d: `M${x} ${y}L${ex} ${ey}`, class: "stalk inh" }));
    svg.append(s("path", { d: `M${ex - ux * 3.4} ${ey - uy * 3.4}L${ex + ux * 3.4} ${ey + uy * 3.4}`, class: "bar" + (recurrent ? " rec" : "") }));
  }
}

function buildTree(factors, depth) {
  const node = { depth, children: [], hidden: 0, factor: factors[depth] || 0 };
  if (depth >= factors.length) return node;
  const b = factors[depth];
  const shown = b > MAX_SHOWN ? 3 : b;
  node.hidden = b - shown;
  for (let i = 0; i < shown; i++) node.children.push(buildTree(factors, depth + 1));
  return node;
}

function leaves(node) {
  return node.children.length ? node.children.reduce((a, c) => a + leaves(c), 0) + (node.hidden ? 0.6 : 0) : 1;
}

function layout(node, left, right) {
  if (!node.children.length) {
    node.x = (left + right) / 2;
    return;
  }
  const total = leaves(node);
  let x = left;
  for (const child of node.children) {
    const share = ((right - left) * leaves(child)) / total;
    layout(child, x, x + share);
    x += share;
  }
  node.x = (node.children[0].x + node.children.at(-1).x) / 2;
}

function walk(node, fn) {
  fn(node);
  node.children.forEach(child => walk(child, fn));
}

// Per-level table: compartments and contacts for one neuron and the population.
export function levelTable({ levels, counts, neurons }) {
  const streams = [
    ["ff_excitatory", "E"],
    ["ff_inhibitory", "I"],
    ["rec_excitatory", "rec E"],
    ["rec_inhibitory", "rec I"],
  ].filter(([key]) => (counts[key] || []).some(Boolean) || key.startsWith("ff"));
  const table = h("table", { class: "levels" });
  table.append(
    h("caption", { text: "Compartments per level and contacts on each compartment" }),
    h("thead", {}, [h("tr", {}, [h("th", { text: "Level" }), h("th", { title: "Compartments at this level", text: "Comp." }), ...streams.map(([, label]) => h("th", { title: `${label} contacts on each compartment`, text: label }))])])
  );
  const body = h("tbody");
  const deepest = levels.length - 1;
  const totals = Object.fromEntries(streams.map(([k]) => [k, 0]));
  for (let depth = deepest; depth >= 0; depth--) {
    const row = h("tr", {}, [h("td", { text: LEVEL_NAMES(depth, deepest) }), h("td", { text: fmt(levels[depth]) })]);
    for (const [key] of streams) {
      const v = (counts[key] || [])[depth] || 0;
      totals[key] += v * levels[depth];
      row.append(h("td", { text: fmt(v) }));
    }
    body.append(row);
  }
  table.append(body);
  const perNeuron = Object.values(totals).reduce((a, b) => a + b, 0);
  table.append(
    h("tfoot", {}, [
      h("tr", {}, [h("th", { text: "Per neuron" }), h("th", { text: fmt(levels.reduce((a, b) => a + b, 0)) }), ...streams.map(([k]) => h("th", { text: fmt(totals[k]) }))]),
      h("tr", {}, [h("th", { text: `Population (${fmt(neurons)})`, colspan: 2 }), h("th", { colspan: streams.length, text: `${fmt(perNeuron * neurons)} contacts` })]),
    ])
  );
  return table;
}

export function neuronLegend() {
  const item = (svgChildren, label) => h("span", { class: "legend-item" }, [(() => {
    const g = s("svg", { width: 22, height: 14, viewBox: "0 0 22 14", "aria-hidden": "true" }, svgChildren);
    return g;
  })(), label]);
  return h("div", { class: "legend" }, [
    item([s("path", { d: "M2 11H20", class: "dendrite" }), s("path", { d: "M8 11V5", class: "stalk exc" }), s("circle", { cx: 8, cy: 3.5, r: 2.4, class: "bouton" })], "excitatory contact"),
    item([s("path", { d: "M2 11H20", class: "dendrite" }), s("path", { d: "M12 11V5", class: "stalk inh" }), s("path", { d: "M8.6 4.5H15.4", class: "bar" })], "inhibitory contact"),
    item([s("path", { d: "M5 13L11 2L17 13Z", class: "soma exc" })], "excitatory soma"),
    item([s("circle", { cx: 11, cy: 7.5, r: 5.5, class: "soma inh" })], "inhibitory soma"),
  ]);
}

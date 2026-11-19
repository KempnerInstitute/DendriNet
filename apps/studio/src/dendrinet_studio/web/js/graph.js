// The population-network canvas: layer lanes, population cards with ports,
// and connections that enter a card on its left (input) side and leave it on
// its right (output) side. Excitatory connections end in an arrow, inhibitory
// ones in a bar; delayed connections are dashed and labelled Δt.

import { s, fmt } from "./dom.js";

const CARD_W = 204;
const CARD_H = 92;
const GAP_Y = 26;
const LANE_PAD = 26;
const HEADER = 44;
const IO_W = 156;
const IO_H = 90;
const SUB_GAP = 34;

const STREAM_LABEL = { ff_excitatory: "E", ff_inhibitory: "I", rec_excitatory: "rec E", rec_inhibitory: "rec I" };

function ruleOf(node) {
  if (node.settings.use_shunting) return ["Shunting", "shunt"];
  const mode = node.settings.additive_mode;
  return [mode && mode !== "raw" ? `Additive (${mode.replace(/_/g, " ")})` : "Additive", "add"];
}

function contactSummary(node) {
  const per = node.configured_per_level;
  const uniform = key => per[key].every(v => v === per[key][0]);
  const part = key => {
    const values = node.received_per_level[key];
    if (!values.some(Boolean)) return null;
    return `${STREAM_LABEL[key]} ${uniform(key) ? per[key][0] : "by level"}`;
  };
  const ff = ["ff_excitatory", "ff_inhibitory"].map(part).filter(Boolean);
  const rec = ["rec_excitatory", "rec_inhibitory"].map(part).filter(Boolean);
  return [ff.length ? ff.join(" · ") + " per comp." : "no feedforward contacts", rec.length ? rec.join(" · ") + " per comp." : null];
}

// Layout: input lane, one lane per layer (inhibitory column left of and above
// the excitatory column, so input reaching the excitatory cells passes below
// the inhibitory ones and local inhibition points forward), decoder lane.
const STAGGER = (CARD_H + GAP_Y) / 2;

export function layoutNetwork(preview) {
  const lanes = [];
  const positions = {};
  let x = 16;
  lanes.push({ kind: "input", label: "Input", x, width: IO_W + 2 * LANE_PAD });
  x += IO_W + 2 * LANE_PAD + 18;
  let stackHeight = CARD_H;
  preview.layers.forEach((layer, li) => {
    const nodes = preview.nodes.filter(n => n.layer === li);
    const inh = nodes.filter(n => n.polarity === "inhibitory");
    const exc = nodes.filter(n => n.polarity !== "inhibitory");
    const columns = [inh, exc].filter(c => c.length);
    const both = columns.length === 2;
    const width = columns.length * CARD_W + (columns.length - 1) * SUB_GAP + 2 * LANE_PAD;
    lanes.push({ kind: "layer", label: layer.name, layer, x, width });
    columns.forEach((column, ci) => {
      const cx = x + LANE_PAD + ci * (CARD_W + SUB_GAP);
      const offset = both ? (ci === 0 ? -STAGGER : STAGGER) : 0;
      column.forEach((node, ri) => {
        positions[node.id] = { x: cx, y: 0, row: ri, count: column.length, offset };
      });
      const columnHeight = column.length * CARD_H + (column.length - 1) * GAP_Y;
      stackHeight = Math.max(stackHeight, columnHeight + (both ? 2 * STAGGER : 0));
    });
    x += width + 18;
  });
  lanes.push({ kind: "output", label: "Readout", x, width: IO_W + 2 * LANE_PAD });
  const width = x + IO_W + 2 * LANE_PAD + 16;
  const height = Math.max(300, HEADER + 30 + stackHeight + 60);
  const top = HEADER + 20 + (height - HEADER - 50 - stackHeight) / 2;
  const mid = top + stackHeight / 2;
  for (const node of preview.nodes) {
    const p = positions[node.id];
    const columnHeight = p.count * CARD_H + (p.count - 1) * GAP_Y;
    p.y = mid - columnHeight / 2 + p.offset + p.row * (CARD_H + GAP_Y);
  }
  positions.__input = { x: lanes[0].x + LANE_PAD, y: mid - IO_H / 2 };
  positions.__output = { x: lanes.at(-1).x + LANE_PAD, y: mid - IO_H / 2 };
  return { lanes, positions, width, height };
}

// The contacts per compartment a connection supplies, from its target's counts.
function edgeCountLabel(preview, edge) {
  const node = preview.nodes.find(n => n.id === edge.target_id);
  const short = STREAM_LABEL[edge.pathway] || edge.pathway;
  if (!node || !node.received_per_level[edge.pathway]) return short;
  const values = [...new Set(node.received_per_level[edge.pathway].filter(v => v > 0))];
  if (!values.length) return `${short} 0`;
  return values.length === 1 ? `${short} ${values[0]}/comp` : `${short} by level`;
}

export function drawNetwork(svg, preview, { selection, onSelectPopulation, onSelectConnection, onSelectLayer, showCounts = true, zoom = 1 }) {
  svg.replaceChildren();
  const { lanes, positions, width, height } = layoutNetwork(preview);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", Math.round(width * zoom));
  svg.setAttribute("height", Math.round(height * zoom));
  const defs = s("defs");
  // Fixed-size markers: a selected (thicker) connection keeps the same arrow.
  defs.append(
    s("marker", { id: "m-exc", viewBox: "0 0 10 10", refX: 9.5, refY: 5, markerWidth: 11, markerHeight: 11, markerUnits: "userSpaceOnUse", orient: "auto-start-reverse" }, [s("path", { d: "M0 0L10 5L0 10Z", class: "marker-exc" })]),
    s("marker", { id: "m-inh", viewBox: "0 0 6 14", refX: 4, refY: 7, markerWidth: 6, markerHeight: 15, markerUnits: "userSpaceOnUse", orient: "auto" }, [s("path", { d: "M4 0V14", class: "marker-inh" })])
  );
  svg.append(defs);

  // Lanes.
  for (const lane of lanes) {
    const g = s("g", { class: "lane" + (lane.kind === "layer" ? " layer-lane" : "") });
    g.append(s("rect", { x: lane.x, y: 8, width: lane.width, height: height - 16, rx: 10, class: "lane-bg" }));
    const label = s("text", { x: lane.x + 14, y: 32, class: "lane-label" }, [lane.label]);
    g.append(label);
    if (lane.kind === "layer") {
      const extras = [];
      if (lane.layer.recurrent) extras.push(`recurrent · dt ${lane.layer.dt}`);
      if (lane.layer.is_readout_layer) extras.push("readout layer");
      if (extras.length) g.append(s("text", { x: lane.x + lane.width - 14, y: 32, class: "lane-meta", "text-anchor": "end" }, [extras.join(" · ")]));
      const index = preview.layers.indexOf(lane.layer);
      g.classList.add("clickable");
      g.addEventListener("click", event => {
        if (event.target === label || event.target.classList.contains("lane-bg")) onSelectLayer(index);
      });
    }
    svg.append(g);
  }

  // Ports: incoming connections spread over the left side, excitatory above inhibitory.
  const incoming = {};
  preview.edges.forEach((edge, index) => {
    (incoming[edge.target_id] ||= []).push({ edge, index });
  });
  const portY = {};
  for (const [target, list] of Object.entries(incoming)) {
    const p = positions[target];
    const excit = list.filter(e => !e.edge.pathway.endsWith("inhibitory"));
    const inhib = list.filter(e => e.edge.pathway.endsWith("inhibitory"));
    excit.forEach((e, i) => (portY[e.index] = p.y + 18 + ((i + 0.5) * (CARD_H / 2 - 18)) / excit.length));
    inhib.forEach((e, i) => (portY[e.index] = p.y + CARD_H / 2 + 4 + ((i + 0.5) * (CARD_H / 2 - 10)) / inhib.length));
  }

  const edgeLayer = s("g", { class: "edges" });
  const chipLayer = s("g", { class: "edge-chips" });
  svg.append(edgeLayer);
  preview.edges.forEach((edge, index) => {
    const external = !positions[edge.source_id];
    const src = external ? positions.__input : positions[edge.source_id];
    const dst = positions[edge.target_id];
    const sx = src.x + (external ? IO_W : CARD_W);
    const sy = external ? src.y + IO_H / 2 + (edge.source_id === "input_i" ? 14 : edge.source_id === "input_e" ? -14 : 0) : src.y + CARD_H / 2;
    const inhibitory = edge.pathway.endsWith("inhibitory");
    const tx = dst.x - (inhibitory ? 4 : 2);
    const ty = portY[index];
    let d;
    if (edge.source_id === edge.target_id) {
      const left = dst.x + 30;
      const right = dst.x + CARD_W - 30;
      d = `M${right} ${dst.y} C ${right} ${dst.y - 44}, ${left} ${dst.y - 44}, ${left} ${dst.y - 2}`;
    } else if (tx > sx + 20) {
      const bend = Math.max(40, (tx - sx) / 2);
      d = `M${sx} ${sy} C ${sx + bend} ${sy}, ${tx - bend} ${ty}, ${tx} ${ty}`;
    } else {
      const over = Math.min(src.y, dst.y) - 26;
      d = `M${sx} ${sy} C ${sx + 60} ${sy}, ${sx + 60} ${over}, ${(sx + tx) / 2} ${over} S ${tx - 60} ${ty}, ${tx} ${ty}`;
    }
    const classes = ["edge", inhibitory ? "inh" : "exc"];
    if (edge.timing === "delayed") classes.push("delayed");
    if (edge.enabled === false) classes.push("disabled");
    const selected = selection.kind === "connection" && selection.layer === edge.layer && selection.index === layerIndex(preview, edge, index);
    if (selected) classes.push("selected");
    const path = s("path", { d, class: classes.join(" "), "data-source": edge.source_id, "data-target": edge.target_id, "data-pathway": edge.pathway, "marker-end": edge.source_id === edge.target_id ? undefined : inhibitory ? "url(#m-inh)" : "url(#m-exc)" });
    const hit = s("path", { d, class: "edge-hit", tabindex: 0, role: "button", "aria-label": `Connection ${edge.source_id} to ${edge.target_id}` });
    const pick = () => onSelectConnection(edge.layer, layerIndex(preview, edge, index));
    hit.addEventListener("click", pick);
    hit.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        pick();
      }
    });
    hit.append(s("title", {}, [`${edge.source_id} → ${edge.target_id} · ${edge.pathway.replace("_", " ")} · ${edge.timing === "delayed" ? "delayed" : "same step"}${edge.target_levels ? " · levels " + edge.target_levels.join(", ") : ""}${edge.probability != null ? " · p " + edge.probability : ""}${edge.enabled === false ? " · off" : ""}`]));
    edgeLayer.append(path, hit);
    // Chip at the middle of the curve.
    const labels = [];
    if (edge.timing === "delayed") labels.push("Δt");
    if (edge.enabled === false) labels.push("off");
    if (edge.target_levels) labels.push("→ " + edge.target_levels.join(","));
    if (edge.probability != null) labels.push("p " + edge.probability);
    if (showCounts || labels.length) {
      requestAnimationFrame(() => {
        if (!path.isConnected) return;
        const length = path.getTotalLength();
        const point = path.getPointAtLength(length * 0.5);
        const text = [showCounts ? edgeCountLabel(preview, edge) : null, ...labels].filter(Boolean).join(" · ");
        const w = text.length * 6.2 + 14;
        const chip = s("g", { class: "chip-edge " + (inhibitory ? "inh" : "exc") + (selected ? " selected" : "") });
        chip.append(s("rect", { x: point.x - w / 2, y: point.y - 9, width: w, height: 18, rx: 9 }), s("text", { x: point.x, y: point.y + 4, "text-anchor": "middle" }, [text]));
        chip.addEventListener("click", pick);
        chipLayer.append(chip);
      });
    }
  });

  // Input and decoder blocks.
  const net = preview.resolved_config.model.core.population_network;
  const inputLines = [net.input_dim ? `${fmt(net.input_dim)} features` : "width from the data", transformText(net.input_transform, net.input_dim), preview.recurrent ? "one vector per time step" : null];
  ioCard(svg, positions.__input, "Input", inputLines);
  const decoder = preview.resolved_config.model.decoder;
  const dp = decoder.params || {};
  const hidden = (dp.hidden_dims || []).length ? `[${dp.hidden_dims.join(", ")}] → ` : "";
  ioCard(svg, positions.__output, decoder.type === "MLP" ? "Decoder (MLP)" : `Decoder (${decoder.type})`, [`${hidden}${dp.output_dim ?? "?"} ${preview.task === "regression" ? "outputs" : "classes"}`, preview.task === "regression" ? "regression" : "classification"]);
  // Readout connection to the decoder.
  const readoutLayer = preview.layers.find(l => l.is_readout_layer) || preview.layers.at(-1);
  const readoutNode = preview.nodes.find(n => n.layer_name === readoutLayer.name && n.name === readoutLayer.readout);
  if (readoutNode) {
    const p = positions[readoutNode.id];
    const o = positions.__output;
    const sx = p.x + CARD_W;
    const sy = p.y + CARD_H / 2;
    const tx = o.x - 2;
    const ty = o.y + IO_H / 2;
    edgeLayer.append(s("path", { d: `M${sx} ${sy} C ${sx + 50} ${sy}, ${tx - 50} ${ty}, ${tx} ${ty}`, class: "edge readout", "marker-end": "url(#m-exc)" }));
  }

  // Population cards.
  for (const node of preview.nodes) {
    const p = positions[node.id];
    const selected = selection.kind === "population" && selection.layer === node.layer && selection.population === node.population;
    const g = s("g", { class: "card " + (node.polarity === "inhibitory" ? "inh" : "exc") + (selected ? " selected" : ""), tabindex: 0, role: "button", "aria-label": `Edit population ${node.layer_name}.${node.name}`, transform: `translate(${p.x},${p.y})` });
    g.append(s("rect", { width: CARD_W, height: CARD_H, rx: 10, class: "card-bg" }));
    if (node.polarity === "inhibitory") g.append(s("circle", { cx: 20, cy: 21, r: 6.5, class: "card-glyph" }));
    else g.append(s("path", { d: "M20 13.5L27 27H13Z", class: "card-glyph" }));
    g.append(s("text", { x: 36, y: 26, class: "card-title" }, [node.name]));
    const [rule, cls] = ruleOf(node);
    const tw = rule.length * 6.1 + 14;
    g.append(s("rect", { x: CARD_W - 10 - tw, y: 12, width: tw, height: 18, rx: 4, class: "rule-tag " + cls }));
    g.append(s("text", { x: CARD_W - 10 - tw / 2, y: 25, "text-anchor": "middle", class: "rule-text " + cls }, [rule]));
    g.append(s("text", { x: 14, y: 50, class: "card-line" }, [`${node.polarity === "inhibitory" ? "I" : "E"} · ${fmt(node.neurons)} neurons · ${node.branch_factors.length ? "[" + node.branch_factors.join(", ") + "]" : "soma only"}`]));
    const [ff, rec] = contactSummary(node);
    g.append(s("text", { x: 14, y: 68, class: "card-line mono" }, [ff]));
    if (rec) g.append(s("text", { x: 14, y: 84, class: "card-line mono" }, [rec]));
    if (readoutNode && node.id === readoutNode.id) g.append(s("text", { x: CARD_W - 10, y: CARD_H - 8, "text-anchor": "end", class: "readout-tag" }, ["readout ▸"]));
    const pick = () => onSelectPopulation(node.layer, node.population);
    g.addEventListener("click", pick);
    g.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        pick();
      }
    });
    svg.append(g);
  }
  svg.append(chipLayer);
  return { width, height };
}

function layerIndex(preview, edge, globalIndex) {
  // Index of the connection within its layer's list, as written in the YAML.
  return preview.edges.slice(0, globalIndex).filter(e => e.layer === edge.layer).length;
}

function transformText(transform, dim) {
  if (transform === "signed_split") return dim ? `ON/OFF split → ${fmt(2 * dim)} channels` : "ON/OFF split";
  if (transform === "nonnegative" || transform === "relu") return `${transform} input`;
  return "used as given";
}

function ioCard(svg, p, title, lines) {
  const g = s("g", { class: "io-card", transform: `translate(${p.x},${p.y})` });
  g.append(s("rect", { width: IO_W, height: IO_H, rx: 10, class: "io-bg" }));
  g.append(s("text", { x: 14, y: 26, class: "card-title" }, [title]));
  lines.filter(Boolean).forEach((line, i) => g.append(s("text", { x: 14, y: 46 + 16 * i, class: "card-line" }, [line])));
  svg.append(g);
}

// Block diagram for model families without a population graph.
export function drawOverview(svg, preview) {
  svg.replaceChildren();
  const o = preview.overview || {};
  const st = o.structured || {};
  const coreLines = [];
  if (st.excitatory_layer_sizes?.length) {
    coreLines.push(`E layers ${st.excitatory_layer_sizes.join(", ")} · I ${st.inhibitory_layer_sizes.join(", ") || "—"}`);
    if (st.branch_factors.length) coreLines.push(`branching [${st.branch_factors.join(", ")}]`);
    if (st.ee_per_branch.length) coreLines.push(`E ${st.ee_per_branch.join("/")} · I ${st.ie_per_branch.join("/")} per branch`);
  } else {
    coreLines.push(preview.recurrent ? "recurrent core" : "feedforward core");
  }
  const blocks = [
    ["Input", [o.input_dim ? `${fmt(o.input_dim)} features` : "from the dataset", preview.dataset]],
    [`Encoder`, [o.encoder || "identity"]],
    [preview.family, coreLines],
    ["Decoder", [`${o.decoder || "MLP"}${o.decoder_hidden?.length ? " [" + o.decoder_hidden.join(", ") + "]" : ""}`, `${o.output_dim ?? "?"} ${preview.task === "regression" ? "outputs" : "classes"}`]],
  ];
  const bw = 196;
  const gap = 46;
  const width = 16 + blocks.length * bw + (blocks.length - 1) * gap + 16;
  const height = 200;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  const defs = s("defs", {}, [s("marker", { id: "m-ov", viewBox: "0 0 10 10", refX: 9.5, refY: 5, markerWidth: 8, markerHeight: 8, orient: "auto" }, [s("path", { d: "M0 0L10 5L0 10Z", class: "marker-exc" })])]);
  svg.append(defs);
  blocks.forEach(([title, lines], i) => {
    const x = 16 + i * (bw + gap);
    const y = 50;
    const g = s("g", { class: i === 2 ? "card exc selected" : "io-card", transform: `translate(${x},${y})` });
    g.append(s("rect", { width: bw, height: 100, rx: 10, class: i === 2 ? "card-bg" : "io-bg" }));
    g.append(s("text", { x: 14, y: 28, class: "card-title" }, [title]));
    lines.filter(Boolean).forEach((line, k) => g.append(s("text", { x: 14, y: 50 + 17 * k, class: "card-line" }, [String(line)])));
    svg.append(g);
    if (i < blocks.length - 1) svg.append(s("path", { d: `M${x + bw + 4} ${y + 50}H${x + bw + gap - 4}`, class: "edge exc", "marker-end": "url(#m-ov)" }));
  });
}

// Export a drawing as a standalone SVG file with its colours resolved.
export function svgForExport(svg) {
  const clone = svg.cloneNode(true);
  const css = getComputedStyle(document.documentElement);
  const tokens = ["--ink", "--muted", "--line", "--line-strong", "--panel", "--sunk", "--exc", "--inh", "--shunt", "--add", "--edge"];
  const vars = tokens.map(t => `${t}: ${css.getPropertyValue(t).trim()};`).join(" ");
  const sheet = [...document.styleSheets].flatMap(sheet => {
    try {
      return [...sheet.cssRules].map(r => r.cssText).filter(text => /\.(card|lane|edge|chip-edge|io-|rule-|marker|dendrite|stalk|bouton|bar|soma|level-|count-label|more|readout)/.test(text));
    } catch (error) {
      return [];
    }
  });
  const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
  style.textContent = `svg { ${vars} font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; } ${sheet.join("\n")}`;
  clone.insertBefore(style, clone.firstChild);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  return new XMLSerializer().serializeToString(clone);
}

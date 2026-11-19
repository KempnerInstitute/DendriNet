// The Model view: outline, network canvas, neuron and inspector.
//
// Population networks are edited visually (populations, connections, layers,
// morphology and per-level contacts). Every other model family is edited
// through forms generated from the library's typed configuration, with a block
// diagram of encoder, core and decoder.

import { h, s, icon, fmt, polarityGlyph, download } from "./dom.js";
import { state, api, edit, select, adopt, emit, getPath, deletePath, isPopulationNetwork, layers, nodeView, configKey } from "./store.js";
import { field, textInput, numberInput, stepper, selectInput, switchInput, segmented, mappingInput, factorEditor, sectionForm, disclosure, humanize, controlFor, describeCount, rebuildKeepingFocus } from "./fields.js";
import { drawNetwork, drawOverview, layoutNetwork, svgForExport } from "./graph.js";
import { externalControls, sourcesTable, structuredInputGuide } from "./afferents.js";
import { drawNeuron, levelTable, neuronLegend } from "./morphology.js";
import { toast, confirmDialog } from "./ui.js";

const NET = "model.core.population_network";
const STREAMS = [
  ["ff_excitatory", "Excitatory", "same step"],
  ["ff_inhibitory", "Inhibitory", "same step"],
  ["rec_excitatory", "Recurrent excitatory", "delayed"],
  ["rec_inhibitory", "Recurrent inhibitory", "delayed"],
];
const PAYLOAD_KEYS = ["population_network", "unified_ei", "baseline_rnn", "heterogeneous_leak_ctrnn", "legendre_memory", "feature_readout", "routed_ei", "shared_projection_shunting", "affine_mlp", "vision"];
const CORE_SECTIONS = ["architecture", "connectivity", "morphology", "transfer", "sparsity", "reactivation", "blocklinear", "implementation", "spatial"];
const POP_GROUPS = {
  Integration: ["use_shunting", "additive_mode", "leak_conductance", "learnable_leak", "use_additive_normalization"],
  "Dendritic gate": ["reactivate", "reactivation_type", "reactivation_init_policy", "reactivation_init_m", "reactivation_init_b"],
  Initialization: ["topk_init_method", "weight_transform", "dbl_init_method", "initialization_seed", "adaptive_initialization"],
  "Time constants": ["tau_base", "tau_ratio", "learnable_tau", "level_taus"],
};
const CURATED = new Set([...Object.values(POP_GROUPS).flat(), ...STREAMS.flatMap(([k]) => [`${k}_synapses`, `${k}_synapses_by_level`]), "somatic_synapses"]);

let els = null;
let zoom = 1;
let zoomMode = "fit";
let lastWidth = 0;
let natural = { width: 800, height: 300 };
let showCounts = true;

export const familyLabel = type => {
  const known = { population_network: "Population network", gru: "GRU", lstm: "LSTM", vanilla_rnn: "Vanilla RNN", rnn: "Vanilla RNN", heterogeneous_leak_ctrnn: "Multi-timescale CTRNN", legendre_memory: "Legendre memory", unified_ei: "Unified E/I network", mlp: "MLP" };
  return known[type] || (type ? type.replace(/_/g, " ") : "model");
};

function popSpec(name) {
  return state.schema.payloads.population?.fields.find(f => f.name === name);
}

// Layout ------------------------------------------------------------------------

export function mountModel(container) {
  els = {
    outline: h("aside", { class: "outline", "aria-label": "Model outline" }),
    banner: h("div", { class: "notice error banner", hidden: true, role: "alert" }),
    summary: h("div", { class: "summary-strip", "aria-label": "Model summary" }),
    canvas: h("section", { class: "panel canvas-panel" }),
    detail: h("section", { class: "panel detail-panel" }),
    inspector: h("aside", { class: "inspector", "aria-label": "Inspector" }),
  };
  container.replaceChildren(h("div", { class: "model-layout" }, [els.outline, h("div", { class: "center" }, [els.banner, els.summary, els.canvas, els.detail]), els.inspector]));
  buildCanvasPanel();
  renderModel();
}

export function renderModel() {
  if (!els || !state.preview) return;
  renderOutline();
  renderBanner();
  renderSummary();
  renderCanvas();
  rebuildKeepingFocus(els.detail, renderDetail);
  rebuildKeepingFocus(els.inspector, renderInspector);
}

function renderBanner() {
  els.banner.hidden = !state.error;
  if (!state.error) return;
  els.banner.replaceChildren(
    icon("cross"),
    h("span", { text: `The last change is not valid: ${state.error}` }),
    h("span", { class: "muted", text: " The drawing shows the last valid design." }),
    h("span", { class: "spacer" }),
    h("button", { type: "button", class: "btn small", onclick: () => emit("undo") }, [icon("undo"), "Undo"])
  );
}

// Outline -------------------------------------------------------------------------

export async function openExample(example) {
  if (state.dirty && !(await confirmDialog({ title: "Open an example", text: `Replace the current design with “${example.title}”? Undo brings the current design back.`, confirm: "Open example" }))) return;
  try {
    const result = await api(`example/${encodeURIComponent(example.name)}`);
    adopt(result, `${example.name}.yaml`, { keepHistory: state.dirty });
    emit("show-view", "model");
  } catch (error) {
    toast(error.message, { kind: "error" });
  }
}

function renderOutline() {
  const p = state.preview;
  const parts = [h("h2", { class: "eyebrow", text: "Start from" })];
  const list = h("div", { class: "examples" });
  for (const ex of state.schema.examples) {
    const active = state.docName === `${ex.name}.yaml`;
    list.append(
      h("button", { type: "button", class: "example" + (active ? " active" : ""), "aria-current": active ? "true" : undefined, title: ex.description, onclick: () => openExample(ex) }, [
        h("span", { class: "example-title", text: ex.title }),
        h("span", { class: "example-meta", text: [familyLabel(ex.family), ex.recurrent ? "recurrent" : null, ex.task, ex.download ? "downloads data" : "no download"].filter(Boolean).join(" · ") }),
      ])
    );
  }
  parts.push(list);
  parts.push(...(p.family === "population_network" && isPopulationNetwork() ? populationOutline() : familyOutline()));
  els.outline.replaceChildren(...parts);
}

function treeRow({ current, onclick, cls = "", title }, children) {
  return h("button", { type: "button", class: `tree-row ${cls}`, "aria-current": current ? "true" : undefined, onclick, title }, children);
}

function targetLayer() {
  const sel = state.selection;
  return Number.isInteger(sel.layer) && sel.layer < layers().length ? sel.layer : layers().length - 1;
}

function decoderText() {
  const d = state.preview.resolved_config.model.decoder || {};
  const p = d.params || {};
  const hidden = (p.hidden_dims || []).join(" → ");
  return `Decoder · ${d.type || "?"} ${hidden ? hidden + " → " : ""}${p.output_dim ?? "?"}`;
}

function populationOutline() {
  const p = state.preview;
  const sel = state.selection;
  const tree = h("div", { class: "tree" });
  const width = getPath(p.resolved_config, NET).input_dim;
  tree.append(treeRow({ current: sel.kind === "network", onclick: () => select({ kind: "network" }) }, [icon("network"), h("span", { class: "grow", text: width ? `Input · ${fmt(width)} features` : "Input · width from the data" })]));
  p.layers.forEach((layer, li) => {
    tree.append(
      treeRow({ cls: "layer-row", current: sel.kind === "layer" && sel.layer === li, onclick: () => select({ kind: "layer", layer: li }) }, [
        icon("layers"),
        h("span", { class: "grow", text: layer.name }),
        h("span", { class: "meta", text: [layer.recurrent ? "recurrent" : null, `readout ${layer.readout}`].filter(Boolean).join(" · ") }),
      ])
    );
    for (const node of p.nodes.filter(n => n.layer === li)) {
      const current = sel.kind === "population" && sel.layer === li && sel.population === node.population;
      tree.append(
        treeRow({ cls: "pop-row", current, onclick: () => select({ kind: "population", layer: li, population: node.population }) }, [
          polarityGlyph(node.polarity),
          h("span", { class: "grow", text: `${node.polarity === "inhibitory" ? "I" : "E"} · ${node.name}` }),
          h("span", { class: "meta", text: `${fmt(node.neurons)} · ${node.branch_factors.length ? "[" + node.branch_factors.join(",") + "]" : "soma"}` }),
        ])
      );
    }
    const count = (layers()[li].connections || []).length;
    tree.append(
      treeRow({ cls: "conn-row", current: sel.kind === "connection" && sel.layer === li, onclick: () => (count ? select({ kind: "connection", layer: li, index: 0 }) : addConnection(li)) }, [
        h("span", { class: "indent" }),
        h("span", { class: "grow", text: count ? describeCount(count, "connection") : "Add a connection" }),
      ])
    );
  });
  tree.append(treeRow({ current: sel.kind === "decoder", onclick: () => select({ kind: "decoder" }) }, [icon("sliders"), h("span", { class: "grow", text: decoderText() })]));
  const target = targetLayer();
  return [
    h("h2", { class: "eyebrow", text: "Network" }),
    tree,
    h("div", { class: "add-row" }, [
      h("button", { type: "button", class: "btn small", onclick: () => addPopulation(target, "excitatory") }, [icon("plus"), "E population"]),
      h("button", { type: "button", class: "btn small", onclick: () => addPopulation(target, "inhibitory") }, [icon("plus"), "I population"]),
      h("button", { type: "button", class: "btn small", onclick: () => addConnection(target) }, [icon("plus"), "Connection"]),
      h("button", { type: "button", class: "btn small", onclick: addLayer }, [icon("plus"), "Layer"]),
    ]),
    h("p", { class: "help", text: `New populations and connections go into ${state.preview.layers[target]?.name}.` }),
  ];
}

function familyOutline() {
  const p = state.preview;
  const sel = state.selection;
  const tree = h("div", { class: "tree" }, [
    treeRow({ current: sel.kind === "encoder", onclick: () => select({ kind: "encoder" }) }, [icon("network"), h("span", { class: "grow", text: `Encoder · ${p.overview?.encoder || "identity"}` })]),
    treeRow({ current: sel.kind === "model", onclick: () => select({ kind: "model" }) }, [icon("layers"), h("span", { class: "grow", text: familyLabel(p.family) }), h("span", { class: "meta", text: p.recurrent ? "recurrent" : "feedforward" })]),
    treeRow({ current: sel.kind === "decoder", onclick: () => select({ kind: "decoder" }) }, [icon("sliders"), h("span", { class: "grow", text: decoderText() })]),
  ]);
  return [h("h2", { class: "eyebrow", text: "Model" }), tree];
}

// Summary -------------------------------------------------------------------------

export function currentCheck() {
  const check = state.modelCheck;
  return check && check.key === configKey() && check.resources ? check : null;
}

function renderSummary() {
  const p = state.preview;
  const tile = (value, label, source) => h("div", { class: "tile" }, [h("div", { class: "tile-value", text: value }), h("div", { class: "tile-label", text: label }), h("div", { class: "tile-source", text: source })]);
  const tiles = [];
  if (p.family === "population_network") {
    tiles.push(tile(fmt(p.layers.length), p.layers.length === 1 ? "layer" : "layers", "configuration"));
    tiles.push(tile(fmt(p.nodes.length), p.nodes.length === 1 ? "population" : "populations", "configuration"));
    tiles.push(tile(fmt(p.nodes.reduce((a, n) => a + n.neurons, 0)), "neurons", "configuration"));
    tiles.push(tile(fmt(p.compartments_total), "compartments incl. soma", "configuration"));
    tiles.push(tile(fmt(p.synapses_total), "synaptic contacts", "configuration"));
  } else {
    tiles.push(tile(familyLabel(p.family), "model family", "configuration"));
    tiles.push(tile(p.recurrent ? "Recurrent" : "Feedforward", "dynamics", "configuration"));
    tiles.push(tile(p.task === "regression" ? "Regression" : "Classification", "task", "configuration"));
  }
  const check = currentCheck();
  if (check) {
    tiles.push(tile(fmt(check.resources.active_synapses), "active synapses", "model check"));
    tiles.push(tile(fmt(check.resources.trainable_parameters), "trainable parameters", "model check"));
  } else {
    const pending = state.modelCheck && state.modelCheck.key === configKey() && !state.modelCheck.resources;
    tiles.push(tile("—", "trainable parameters", pending ? (state.modelCheck.error ? "model check failed" : "checking…") : "run a model check"));
  }
  els.summary.replaceChildren(...tiles);
}

// Canvas --------------------------------------------------------------------------

function iconButton(name, label, onclick) {
  return h("button", { type: "button", class: "icon-btn", "aria-label": label, title: label, onclick }, [icon(name)]);
}

function buildCanvasPanel() {
  const counts = h("input", { type: "checkbox", id: "show-counts" });
  counts.checked = showCounts;
  counts.addEventListener("change", () => {
    showCounts = counts.checked;
    renderCanvas();
  });
  els.svg = s("svg", { class: "network", role: "group", "aria-label": "Network diagram" });
  els.scroller = h("div", { class: "canvas-scroll" }, [els.svg]);
  els.canvasTitle = h("h3", { text: "Network" });
  els.canvasSub = h("span", { class: "sub" });
  els.canvasTools = h("div", { class: "tools" }, [
    h("label", { class: "check", for: "show-counts" }, [counts, "Contact counts"]),
    iconButton("zoomOut", "Zoom out", () => setZoom(zoom / 1.2)),
    iconButton("zoomIn", "Zoom in", () => setZoom(zoom * 1.2)),
    iconButton("fit", "Fit to the panel", fitZoom),
    iconButton("download", "Export the diagram (SVG)", exportSvg),
  ]);
  els.legend = h("div", { class: "legend" });
  if (window.ResizeObserver) {
    new ResizeObserver(() => {
      const width = els.scroller.clientWidth;
      if (zoomMode === "fit" && Math.abs(width - lastWidth) > 8 && state.preview?.family === "population_network") {
        lastWidth = width;
        renderCanvas();
      }
    }).observe(els.scroller);
  }
  els.canvas.replaceChildren(h("div", { class: "panel-head" }, [els.canvasTitle, els.canvasSub, h("span", { class: "spacer" }), els.canvasTools]), els.scroller, els.legend);
}

function setZoom(value) {
  zoomMode = "manual";
  zoom = Math.min(2.5, Math.max(0.35, value));
  renderCanvas();
}

function fitZoom() {
  zoomMode = "fit";
  renderCanvas();
}

// In fit mode the network fills the panel's width (never enlarged past 110%).
function fittedZoom(width) {
  const available = els.scroller.clientWidth - 12;
  return available > 120 ? Math.min(1.1, Math.max(0.35, available / width)) : zoom;
}

function exportSvg() {
  const name = (state.docName || "dendrinet_model").replace(/\.(ya?ml)$/i, "");
  download(`${name}_network.svg`, svgForExport(els.svg), "image/svg+xml");
}

function renderCanvas() {
  const p = state.preview;
  if (p.family === "population_network") {
    els.canvasTitle.textContent = "Network";
    els.canvasSub.textContent = "configured pathways between populations, not individual synapses";
    els.canvasTools.hidden = false;
    if (zoomMode === "fit") zoom = fittedZoom(layoutNetwork(p).width);
    natural = drawNetwork(els.svg, p, {
      selection: state.selection,
      showCounts,
      zoom,
      onSelectPopulation: (layer, population) => select({ kind: "population", layer, population }),
      onSelectConnection: (layer, index) => select({ kind: "connection", layer, index }),
      onSelectLayer: layer => select({ kind: "layer", layer }),
    });
    els.legend.replaceChildren(...networkLegend());
  } else {
    els.canvasTitle.textContent = "Model";
    els.canvasSub.textContent = "encoder, core and decoder as configured";
    els.canvasTools.hidden = true;
    drawOverview(els.svg, p);
    els.legend.replaceChildren();
  }
}

function legendItem(children, label, width = 26) {
  return h("span", { class: "legend-item" }, [s("svg", { width, height: 14, viewBox: `0 0 ${width} 14`, "aria-hidden": "true" }, children), label]);
}

function networkLegend() {
  return [
    h("span", { class: "legend-item" }, [polarityGlyph("excitatory"), "E population"]),
    h("span", { class: "legend-item" }, [polarityGlyph("inhibitory"), "I population"]),
    legendItem([s("path", { d: "M2 7H19", class: "edge exc" }), s("path", { d: "M18 3L25 7L18 11Z", class: "marker-exc" })], "excitation (arrow)"),
    legendItem([s("path", { d: "M2 7H22", class: "edge inh" }), s("path", { d: "M23 1V13", class: "marker-inh" })], "inhibition (bar end)"),
    legendItem([s("path", { d: "M2 7H24", class: "edge exc delayed" })], "delayed one step (Δt)"),
    legendItem([s("path", { d: "M2 7H24", class: "edge exc disabled" })], "switched off"),
  ];
}

// Neuron / family forms --------------------------------------------------------------

function focusNode() {
  const sel = state.selection;
  const p = state.preview;
  if (sel.kind === "population") return nodeView(sel.layer, sel.population);
  if (sel.kind === "connection") {
    const edge = p.edges.filter(e => e.layer === sel.layer)[sel.index];
    return edge ? p.nodes.find(n => n.id === edge.target_id) : null;
  }
  const layerIndex = sel.kind === "layer" ? sel.layer : p.layers.findIndex(l => l.is_readout_layer);
  const layer = p.layers[layerIndex];
  return layer ? p.nodes.find(n => n.layer === layerIndex && n.name === layer.readout) : p.nodes[0];
}

function renderDetail() {
  const p = state.preview;
  if (p.family !== "population_network") {
    renderFamilyForms();
    return;
  }
  const node = focusNode();
  if (!node) {
    els.detail.replaceChildren();
    return;
  }
  const sel = state.selection;
  let highlight = null;
  if (sel.kind === "connection") {
    const edge = p.edges.filter(e => e.layer === sel.layer)[sel.index];
    highlight = edge?.target_depths || null;
  }
  const svg = s("svg", { class: "neuron", role: "img" });
  drawNeuron(svg, { polarity: node.polarity, factors: node.branch_factors, counts: node.received_per_level, highlight, title: `Neuron of ${node.layer_name}.${node.name}` });
  const notes = [];
  for (const [key, label] of STREAMS) {
    const configured = node.configured_per_level[key].some(Boolean);
    const received = node.received_per_level[key].some(Boolean);
    if (configured && !received) notes.push(`${label} contacts are configured but no ${label.toLowerCase()} connection reaches this population, so none are built.`);
  }
  if (!node.settings.somatic_synapses) notes.push("The soma receives no synaptic contacts (somatic synapses off).");
  const caption = sel.kind === "connection" && highlight ? `Highlighted: levels this connection reaches.` : null;
  els.detail.replaceChildren(
    h("div", { class: "panel-head" }, [h("h3", { text: "Neuron" }), h("span", { class: "sub", text: `${node.layer_name} · ${node.name}: one of ${fmt(node.neurons)} identical neurons` })]),
    h("div", { class: "neuron-body" }, [
      h("figure", { class: "neuron-figure" }, [svg, neuronLegend(), caption ? h("figcaption", { class: "help", text: caption }) : null]),
      h("div", { class: "neuron-side" }, [
        levelTable({ levels: node.levels_including_soma, counts: node.received_per_level, neurons: node.neurons }),
        sourcesTable(node),
        equationBox(node.settings),
        ...notes.map(text => h("p", { class: "notice warn compact" }, [icon("warn"), h("span", { text })])),
        h("p", { class: "help", text: "The tree is the exact compartment tree; its geometry is schematic. Counts are contacts per compartment, as configured; a model check counts the synapses of the built model." }),
      ]),
    ])
  );
}

function equationBox(st) {
  let title;
  let formula;
  let note = null;
  if (st.use_shunting) {
    title = "Shunting";
    formula = "V = (E + C) / (L + E + I + G + ε)";
    note = `L = ${fmt(st.leak_conductance)}${st.learnable_leak ? ", learned per compartment" : ", fixed"}`;
  } else if (st.additive_mode === "conductance_normalized") {
    title = "Additive, conductance-normalized";
    formula = "V = (E − I + C) / (1 + E + I + G + ε)";
  } else if (st.additive_mode === "tangent_matched") {
    title = "Additive, tangent-matched";
    formula = "V = V₀ + ((N − N₀) − V₀·(T − T₀)) / D₀";
    note = "first-order expansion of unit-leak shunting around (N₀, T₀)";
  } else {
    title = "Additive";
    formula = "V = E − I + C";
    if (st.use_additive_normalization) note = "then standardized across the compartments of the level";
  }
  return h("div", { class: `equation ${st.use_shunting ? "shunt" : "add"}` }, [
    h("div", { class: "eq-title", text: title }),
    h("code", { class: "eq", text: formula }),
    note ? h("div", { class: "eq-note", text: note }) : null,
    h("div", { class: "eq-note", text: "E, I: excitatory and inhibitory synaptic input · C: input from child branches · G: child coupling · output f(V) through the dendritic gate" }),
  ]);
}

function payloadKey() {
  return state.preview.payload;
}

function renderFamilyForms() {
  const p = state.preview;
  const core = state.config.model.core;
  const payload = payloadKey();
  const body = h("div", { class: "panel-body" });
  if (payload) {
    const base = `model.core.${payload}`;
    const typed = state.schema.payloads[payload] || state.schema.sections[base];
    body.append(h("p", { class: "help", text: `${familyLabel(p.family)} reads only ${base}.` }));
    if (typed) body.append(sectionForm(typed, { base, columns: 2, readValue: payloadReader(base, typed) }));
    else {
      body.append(
        field({
          label: `${payload} options (YAML)`,
          control: mappingInput(core[payload] || {}, value => edit(cfg => (cfg.model.core[payload] = value), { path: base, structural: true })),
          key: base,
          wide: true,
          help: "This family's options are a nested structure; edit them as YAML. DendriNet checks them when the configuration is validated.",
        })
      );
    }
  } else {
    body.append(h("p", { class: "help", text: "These sections describe the core. Options a family does not read are rejected when the configuration is checked, so nothing is silently ignored." }));
    const guide = structuredInputGuide(p);
    if (guide) body.append(guide);
    for (const name of CORE_SECTIONS) {
      const section = state.schema.sections[`model.core.${name}`];
      if (!section) continue;
      const set = core[name] && Object.keys(core[name]).length;
      body.append(disclosure(`core-${name}`, humanize(name), () => sectionForm(section, { base: `model.core.${name}`, columns: 2 }), { open: Boolean(set) && ["architecture", "connectivity", "morphology"].includes(name), meta: set ? `${Object.keys(core[name]).length} set` : "defaults" }));
    }
  }
  els.detail.replaceChildren(h("div", { class: "panel-head" }, [h("h3", { text: `${familyLabel(p.family)} options` }), h("span", { class: "sub", text: `model.core${payload ? "." + payload : ""}` })]), body);
}

// Reads a payload option: the configuration's value, else the payload class default.
function payloadReader(base, section) {
  const defaults = Object.fromEntries(section.fields.map(f => [f.name, f.default]));
  return (name, which) => (which === "own" ? getPath(state.config, `${base}.${name}`) : defaults[name]);
}

// Inspector ------------------------------------------------------------------------

function renderInspector() {
  const sel = state.selection;
  const builders = { population: populationInspector, connection: connectionInspector, layer: layerInspector, network: networkInspector, decoder: decoderInspector, encoder: encoderInspector, model: modelInspector };
  const build = builders[sel.kind] || (isPopulationNetwork() ? networkInspector : modelInspector);
  let content;
  try {
    content = build(sel);
  } catch (error) {
    console.error(error);
    content = [h("p", { class: "notice error" }, [icon("cross"), h("span", { text: `This item cannot be shown: ${error.message}` })])];
  }
  els.inspector.classList.toggle("show-keys", state.showKeys);
  els.inspector.replaceChildren(...[].concat(content));
}

function head(glyph, title, sub) {
  const keys = h("input", { type: "checkbox", id: "show-keys" });
  keys.checked = state.showKeys;
  keys.addEventListener("change", () => {
    state.showKeys = keys.checked;
    els.inspector.classList.toggle("show-keys", state.showKeys);
  });
  return h("div", { class: "inspector-head" }, [
    h("div", { class: "inspector-title" }, [glyph, h("h2", { text: title })]),
    sub ? h("p", { class: "sub", text: sub }) : null,
    h("label", { class: "check small", for: "show-keys" }, [keys, "Show YAML keys"]),
  ]);
}

function group(title, children, note) {
  return h("section", { class: "group" }, [h("h3", { class: "eyebrow", text: title }), note ? h("p", { class: "help", text: note }) : null, ...[].concat(children).filter(Boolean)]);
}

// Contexts that read and write per-population options --------------------------------

function populationContext(li, pi) {
  const layer = layers()[li];
  const item = layer.populations[pi];
  const itemPath = `${NET}.layers.${li}.populations.${pi}`;
  const node = nodeView(li, pi);
  return {
    keyOf: name => `${itemPath}.population.${name}`,
    ownOf: name => item.population?.[name],
    inheritedOf: name =>
      layer.population_defaults?.[name] !== undefined ? { value: layer.population_defaults[name], source: "layer default" } : { value: node?.settings?.[name] ?? popSpec(name)?.default, source: "default" },
    write: (name, value) =>
      edit(cfg => {
        const it = getPath(cfg, itemPath);
        it.population = it.population || {};
        it.population[name] = value;
      }, { path: `${itemPath}.population.${name}` }),
    remove: name =>
      edit(cfg => {
        const it = getPath(cfg, itemPath);
        if (!it.population) return;
        delete it.population[name];
        if (!Object.keys(it.population).length) delete it.population;
      }, { path: `${itemPath}.population.${name}`, structural: true }),
  };
}

function layerDefaultsContext(li) {
  const layer = layers()[li];
  const base = `${NET}.layers.${li}.population_defaults`;
  return {
    keyOf: name => `${base}.${name}`,
    ownOf: name => layer.population_defaults?.[name],
    inheritedOf: name => ({ value: popSpec(name)?.default, source: "default" }),
    write: (name, value) =>
      edit(cfg => {
        const l = getPath(cfg, `${NET}.layers.${li}`);
        l.population_defaults = l.population_defaults || {};
        l.population_defaults[name] = value;
      }, { path: `${base}.${name}` }),
    remove: name => edit(cfg => deletePath(cfg, `${base}.${name}`), { path: `${base}.${name}`, structural: true }),
  };
}

function optionField(ctx, name, overrides = {}) {
  const spec = popSpec(name) || { name, kind: "any" };
  const own = ctx.ownOf(name);
  const inherited = ctx.inheritedOf(name);
  const shown = own !== undefined ? own : inherited.value;
  const key = ctx.keyOf(name);
  return field({
    label: overrides.label || humanize(name),
    control: controlFor(spec, shown, value => ctx.write(name, value)),
    key,
    inherited: own === undefined ? inherited.source : false,
    onReset: own !== undefined ? () => ctx.remove(name) : null,
    error: state.errorPath === key ? state.error : null,
    help: overrides.help,
    wide: spec.kind === "mapping" || spec.kind === "any",
  });
}

function integrationGroup(ctx, settings) {
  const own = ctx.ownOf("use_shunting");
  const value = own !== undefined ? own : ctx.inheritedOf("use_shunting").value;
  const children = [
    field({
      label: "Rule",
      control: segmented(Boolean(value), [[true, "Shunting", "shunt"], [false, "Additive", "add"]], v => ctx.write("use_shunting", v), "Integration rule"),
      key: ctx.keyOf("use_shunting"),
      inherited: own === undefined ? ctx.inheritedOf("use_shunting").source : false,
      onReset: own !== undefined ? () => ctx.remove("use_shunting") : null,
      help: value ? "Conductance-like: inhibition and the leak divide the drive." : "Current-like: inhibition is subtracted from the drive.",
    }),
  ];
  if (value) children.push(optionField(ctx, "leak_conductance"), optionField(ctx, "learnable_leak"));
  else children.push(optionField(ctx, "additive_mode", { help: "raw: E − I + C. conductance_normalized: divided by 1 + E + I + G. tangent_matched: the local linearization of shunting." }), optionField(ctx, "use_additive_normalization"));
  if (settings) children.push(equationBox(settings));
  return group("Integration", children);
}

function synapseGroup(ctx, { node, recurrentRelevant, perLevel = true }) {
  const children = [];
  const grid = h("div", { class: "form-grid cols-2" });
  for (const [key, label] of STREAMS) {
    const name = `${key}_synapses`;
    const own = ctx.ownOf(name);
    const shownValue = own !== undefined ? own : ctx.inheritedOf(name).value;
    if (key.startsWith("rec") && !recurrentRelevant && !shownValue) continue;
    const byLevel = ctx.ownOf(`${name}_by_level`) ?? ctx.inheritedOf(`${name}_by_level`).value;
    const reached = !node || node.received_per_level[key].some(Boolean) || !shownValue;
    grid.append(
      field({
        label,
        control: stepper(shownValue ?? 0, v => ctx.write(name, v), { min: 0, label }),
        key: ctx.keyOf(name),
        inherited: own === undefined ? ctx.inheritedOf(name).source : false,
        onReset: own !== undefined ? () => ctx.remove(name) : null,
        error: state.errorPath === ctx.keyOf(name) ? state.error : null,
        help: byLevel?.length ? "Overridden per level below." : node && !reached ? "No connection of this kind reaches this population." : null,
      })
    );
  }
  children.push(grid, optionField(ctx, "somatic_synapses", { label: "Contacts on the soma" }));
  if (perLevel && node) children.push(disclosure(`levels-${ctx.keyOf("x")}`, "Set counts per level", () => perLevelEditor(ctx, node, recurrentRelevant), { meta: "soma → distal" }));
  return group("Synapses per compartment", children, "Contacts on each compartment, per pathway.");
}

function perLevelEditor(ctx, node, recurrentRelevant) {
  const levels = node.levels_including_soma.length;
  const names = depth => (depth === 0 ? "soma" : depth === levels - 1 ? `${depth} · distal` : depth === 1 ? "1 · primary" : String(depth));
  const streams = STREAMS.filter(([key]) => key.startsWith("ff") || recurrentRelevant);
  const table = h("table", { class: "levels editable" });
  table.append(h("thead", {}, [h("tr", {}, [h("th", { text: "Level" }), ...streams.map(([, label]) => h("th", { text: label.replace("Recurrent", "rec.") }))])]));
  const body = h("tbody");
  for (let depth = levels - 1; depth >= 0; depth--) {
    const row = h("tr", {}, [h("th", { scope: "row", text: names(depth) })]);
    for (const [key, label] of streams) {
      const current = node.configured_per_level[key][depth];
      const input = numberInput(current, value => {
        const list = [...node.configured_per_level[key]];
        list[depth] = value;
        ctx.write(`${key}_synapses_by_level`, list.reverse());
      }, { integer: true, min: 0 });
      input.setAttribute("aria-label", `${label} contacts at level ${names(depth)}`);
      input.dataset.key = `${ctx.keyOf(`${key}_synapses_by_level`)}.${depth}`;
      row.append(h("td", {}, [input]));
    }
    body.append(row);
  }
  table.append(body);
  const clearButton = h("button", { type: "button", class: "btn small", onclick: () => {
    for (const [key] of STREAMS) if (ctx.ownOf(`${key}_synapses_by_level`) !== undefined) ctx.remove(`${key}_synapses_by_level`);
  } }, ["Use one count for all levels"]);
  return h("div", {}, [table, h("p", { class: "help", text: "Saved as *_synapses_by_level lists, which run from the most distal level to the soma." }), clearButton]);
}

function groupedOptions(ctx, groupName) {
  return POP_GROUPS[groupName].map(name => optionField(ctx, name));
}

function allOptions(ctx, idPrefix) {
  return disclosure(`${idPrefix}-all`, "All population options", () => {
    const box = h("div");
    const search = h("input", { type: "search", class: "input", placeholder: "Filter options", "aria-label": "Filter population options" });
    const list = h("div", { class: "form-grid cols-1" });
    const fill = () => {
      const term = search.value.trim().toLowerCase();
      const fields = state.schema.payloads.population.fields.filter(f => !CURATED.has(f.name) && (!term || f.name.includes(term) || humanize(f.name).toLowerCase().includes(term)));
      list.replaceChildren(...fields.map(f => optionField(ctx, f.name)));
      if (!fields.length) list.append(h("p", { class: "muted", text: "No matching option." }));
    };
    search.addEventListener("input", fill);
    fill();
    box.append(search, list);
    return box;
  }, { meta: `${state.schema.payloads.population?.fields.length || 0} options` });
}

// Population ---------------------------------------------------------------------------

function populationInspector(sel) {
  const li = sel.layer;
  const pi = sel.population;
  const layer = layers()[li];
  const item = layer.populations[pi];
  const node = nodeView(li, pi);
  const itemPath = `${NET}.layers.${li}.populations.${pi}`;
  const ctx = populationContext(li, pi);
  const polarity = item.polarity || node?.polarity || "excitatory";
  const recurrentRelevant = Boolean(layer.recurrent) || state.preview.edges.some(e => e.target_id === `${layer.name}.${item.name}` && e.timing === "delayed");
  const setItem = (key, value, structural = false) => edit(cfg => (getPath(cfg, itemPath)[key] = value), { path: `${itemPath}.${key}`, structural });
  const factors = item.branch_factors || [];
  const levels = factors.length + 1;
  return [
    head(polarityGlyph(polarity, 18), `${polarity === "inhibitory" ? "Inhibitory" : "Excitatory"} population · ${item.name}`, `in ${layer.name}${node ? ` · ${fmt(node.synapses_per_neuron)} contacts per neuron` : ""}`),
    group("Population", [
      field({ label: "Name", control: textInput(item.name, v => renamePopulation(li, pi, v.trim())), key: `${itemPath}.name` }),
      field({ label: "Cell type", control: segmented(polarity, [["excitatory", "Excitatory", "exc"], ["inhibitory", "Inhibitory", "inh"]], v => setItem("polarity", v, true), "Cell type"), key: `${itemPath}.polarity`, help: "Excitatory populations send excitation; inhibitory populations send inhibition." }),
      field({ label: "Neurons", control: stepper(item.n_neurons, v => setItem("n_neurons", v), { min: 1, label: "neurons" }), key: `${itemPath}.n_neurons`, error: state.errorPath === `${itemPath}.n_neurons` ? state.error : null }),
    ]),
    group(
      "Morphology",
      [
        field({ label: "Branching per level (soma → distal)", control: factorEditor(factors, v => setFactors(li, pi, v)), key: `${itemPath}.branch_factors`, help: `${levels} ${levels === 1 ? "level" : "levels"} · ${fmt(node?.compartments_per_neuron ?? 1)} compartments per neuron` }),
      ]
    ),
    li === 0 && node ? group("External input to this population", externalControls(node), "Choose the incoming synapse signs for this population. Connections from other populations are set separately.") : null,
    synapseGroup(ctx, { node, recurrentRelevant }),
    integrationGroup(ctx, null),
    group("Dendritic gate", groupedOptions(ctx, "Dendritic gate")),
    group("Initialization", groupedOptions(ctx, "Initialization"), "How synaptic weights, couplings and gates start."),
    recurrentRelevant ? group("Time constants", groupedOptions(ctx, "Time constants")) : null,
    allOptions(ctx, `pop-${li}-${pi}`),
    h("div", { class: "danger-zone" }, [h("button", { type: "button", class: "btn danger small", onclick: () => removePopulation(li, pi) }, [icon("trash"), "Remove population"])]),
  ].filter(Boolean);
}

// Connection ---------------------------------------------------------------------------

function connectionInspector(sel) {
  const li = sel.layer;
  const layer = layers()[li];
  const list = layer.connections || [];
  const conn = list[sel.index];
  const path = `${NET}.layers.${li}.connections.${sel.index}`;
  const edge = state.preview.edges.filter(e => e.layer === li)[sel.index];
  const target = edge ? state.preview.nodes.find(n => n.id === edge.target_id) : null;
  const set = (name, value, structural = false) =>
    edit(cfg => {
      const c = getPath(cfg, path);
      if (value === null || value === undefined || value === "") delete c[name];
      else c[name] = value;
    }, { path: `${path}.${name}`, structural });
  const local = layer.populations.map(p => p.name);
  const sources = { Input: li === 0 ? ["input", "input_e", "input_i"] : ["input"], [`Populations in ${layer.name}`]: local };
  const others = layers().flatMap((l, i) => (i === li ? [] : l.populations.map(p => `${l.name}.${p.name}`)));
  if (others.length) sources["Populations of other layers"] = others;
  const sourceLabels = { input: li === 0 ? "input (all features)" : `input (readout of ${layers()[li - 1].name})`, input_e: "input_e (excitatory channels)", input_i: "input_i (inhibitory channels)" };
  const derived = edge ? edge.pathway : "";
  const levels = target ? target.levels_including_soma.length : 1;
  const levelName = depth => (depth === 0 ? "soma" : depth === levels - 1 ? "distal" : depth === 1 ? "primary" : `level ${depth}`);
  const allLevels = conn.target_levels === undefined || conn.target_levels === null;
  const levelBoxes = h("div", { class: "level-checks" });
  for (let depth = levels - 1; depth >= 0; depth--) {
    const box = h("input", { type: "checkbox", id: `lvl-${depth}` });
    box.checked = !allLevels && (edge?.target_depths || []).includes(depth);
    box.disabled = allLevels;
    box.addEventListener("change", () => {
      const chosen = [...levelBoxes.querySelectorAll("input")].filter(b => b.checked).map(b => Number(b.id.slice(4)));
      set("target_levels", chosen.length ? chosen.sort((a, b) => a - b) : null, true);
    });
    levelBoxes.append(h("label", { class: "check", for: `lvl-${depth}` }, [box, `${depth} · ${levelName(depth)}`]));
  }
  const nav = h("div", { class: "nav-row" }, [
    h("button", { type: "button", class: "btn small", disabled: sel.index === 0, onclick: () => select({ kind: "connection", layer: li, index: sel.index - 1 }) }, ["Previous"]),
    h("span", { class: "muted", text: `${sel.index + 1} of ${list.length} in ${layer.name}` }),
    h("button", { type: "button", class: "btn small", disabled: sel.index >= list.length - 1, onclick: () => select({ kind: "connection", layer: li, index: sel.index + 1 }) }, ["Next"]),
  ]);
  const inhibitory = derived.endsWith("inhibitory");
  return [
    head(icon(inhibitory ? "minus" : "chevron"), `Connection · ${conn.source} → ${conn.target}`, edge ? `${derived.replace("_", " ")} · ${edge.timing === "delayed" ? "delayed one step" : "same step"}` : ""),
    nav,
    group("Route", [
      field({ label: "From", control: selectInput(conn.source, [], v => set("source", v, true), { groups: sources, labels: sourceLabels }), key: `${path}.source` }),
      field({ label: "To", control: selectInput(conn.target, local, v => set("target", v, true), { allowCustom: false }), key: `${path}.target` }),
      field({ label: "Timing", control: segmented(conn.timing || "same_step", [["same_step", "Same step"], ["delayed", "Delayed (Δt)"]], v => set("timing", v === "same_step" ? null : v, true), "Timing"), key: `${path}.timing`, help: "Delayed connections carry the source's previous state; they make the network recurrent." }),
      field({
        label: "Pathway",
        control: selectInput(conn.pathway || "", ["", "ff_excitatory", "ff_inhibitory", "rec_excitatory", "rec_inhibitory"], v => set("pathway", v || null, true), { allowCustom: false, labels: { "": `automatic (${edge && !conn.pathway ? derived : "from source and timing"})` } }),
        key: `${path}.pathway`,
        help: "Automatic uses the source's cell type and the timing. The pathway decides which synapse count of the target applies.",
      }),
    ]),
    group("Placement", [
      field({ label: "All levels of the target", control: switchInput(allLevels, v => set("target_levels", v ? null : Array.from({ length: levels }, (_, i) => i), true), "All levels"), key: `${path}.target_levels`, help: allLevels ? null : "Untick the levels this connection should not reach (0 = soma)." }),
      allLevels ? null : levelBoxes,
      field({ label: "Connection probability", control: numberInput(conn.probability ?? null, v => set("probability", v), { optional: true, min: 0, max: 1, step: 0.05 }), key: `${path}.probability`, help: "Empty: every target neuron samples from the whole source." }),
      field({ label: "Seed", control: numberInput(conn.seed ?? null, v => set("seed", v), { integer: true, optional: true }), key: `${path}.seed` }),
      field({ label: "Enabled", control: switchInput(conn.enabled !== false, v => set("enabled", v ? null : false, true), "Enabled"), key: `${path}.enabled`, help: "A switched-off connection stays in the file but is not built." }),
    ]),
    target ? h("p", { class: "help", text: `This source contributes to the ${derived.replace("_", " ")} contact budget of ${target.name} at the selected levels. All sources on this pathway share its budget.` }) : null,
    h("div", { class: "danger-zone" }, [h("button", { type: "button", class: "btn danger small", onclick: () => removeConnection(li, sel.index) }, [icon("trash"), "Remove connection"])]),
  ].filter(Boolean);
}

// Layer ----------------------------------------------------------------------------------

function layerInspector(sel) {
  const li = sel.layer;
  const layer = layers()[li];
  const path = `${NET}.layers.${li}`;
  const view = state.preview.layers[li];
  const set = (name, value, structural = false) =>
    edit(cfg => {
      const l = getPath(cfg, path);
      if (value === null || value === undefined || value === "") delete l[name];
      else l[name] = value;
    }, { path: `${path}.${name}`, structural });
  const ctx = layerDefaultsContext(li);
  const recurrentRelevant = Boolean(layer.recurrent) || (layer.connections || []).some(c => c.timing === "delayed");
  return [
    head(icon("layers"), `Layer · ${layer.name}`, `${describeCount(layer.populations.length, "population")} · ${describeCount((layer.connections || []).length, "connection")}`),
    group("Layer", [
      field({ label: "Name", control: textInput(layer.name, v => renameLayer(li, v.trim())), key: `${path}.name` }),
      field({ label: "Readout population", control: selectInput(layer.readout_population || "", ["", ...layer.populations.filter(p => p.polarity !== "inhibitory").map(p => p.name)], v => set("readout_population", v || null, true), { allowCustom: false, labels: { "": `first excitatory (${view?.readout || "?"})` } }), key: `${path}.readout_population`, help: "Its activity is what the next layer or the decoder reads." }),
      field({ label: "Recurrent", control: switchInput(Boolean(layer.recurrent), v => set("recurrent", v || null, true), "Recurrent"), key: `${path}.recurrent`, help: "Integrate the layer over time steps; required for delayed connections within the layer." }),
      layer.recurrent ? field({ label: "Time step dt", control: numberInput(layer.dt ?? 1, v => set("dt", v), { min: 0 }), key: `${path}.dt` }) : null,
      field({ label: "Connection seed", control: numberInput(layer.connection_seed ?? null, v => set("connection_seed", v), { integer: true, optional: true }), key: `${path}.connection_seed` }),
    ]),
    h("p", { class: "help lead", text: "Defaults for every population in this layer. A population's own setting overrides them." }),
    synapseGroup(ctx, { node: null, recurrentRelevant, perLevel: false }),
    integrationGroup(ctx, null),
    group("Dendritic gate", groupedOptions(ctx, "Dendritic gate")),
    group("Initialization", groupedOptions(ctx, "Initialization")),
    recurrentRelevant ? group("Time constants", groupedOptions(ctx, "Time constants")) : null,
    allOptions(ctx, `layer-${li}`),
    h("div", { class: "add-row" }, [
      h("button", { type: "button", class: "btn small", onclick: () => addPopulation(li, "excitatory") }, [icon("plus"), "E population"]),
      h("button", { type: "button", class: "btn small", onclick: () => addPopulation(li, "inhibitory") }, [icon("plus"), "I population"]),
      h("button", { type: "button", class: "btn small", onclick: () => addConnection(li) }, [icon("plus"), "Connection"]),
    ]),
    layers().length > 1 ? h("div", { class: "danger-zone" }, [h("button", { type: "button", class: "btn danger small", onclick: () => removeLayer(li) }, [icon("trash"), "Remove layer"])]) : null,
  ].filter(Boolean);
}

// Network, encoder, decoder, other families ----------------------------------------------

function networkInspector() {
  const spec = state.schema.payloads.population_network;
  const names = layers().map(l => l.name);
  const base = NET;
  const read = payloadReader(base, spec);
  const net = getPath(state.config, base);
  return [
    head(icon("network"), "Population network", `${describeCount(names.length, "layer")} · input and readout`),
    group("Input", [
      sectionForm(spec, { base, only: ["input_dim", "input_transform", "input_projection_dims", "use_transfer"], readValue: read }),
      h("p", { class: "help", text: "input_transform: identity (default), nonnegative, relu, or signed_split (ON/OFF channels for signed inputs)." }),
    ]),
    group("Readout", [
      field({ label: "Readout layer", control: selectInput(net.readout_layer || "", ["", ...names], v => edit(cfg => (v ? (getPath(cfg, base).readout_layer = v) : delete getPath(cfg, base).readout_layer), { path: `${base}.readout_layer`, structural: true }), { allowCustom: false, labels: { "": `last layer (${names.at(-1)})` } }), key: `${base}.readout_layer` }),
      sectionForm(spec, { base, only: ["output_mode"], readValue: read }),
    ]),
    disclosure("net-all", "All network options", () => sectionForm(spec, { base, exclude: ["layers", "input_dim", "input_transform", "input_projection_dims", "use_transfer", "readout_layer", "output_mode"], readValue: read })),
    encoderGroup(),
  ];
}

function encoderGroup() {
  const section = state.schema.sections["model.encoder.params"];
  return disclosure("encoder", "Encoder", () => h("div", {}, [
    sectionForm(state.schema.sections["model.encoder"], { base: "model.encoder", only: ["type"] }),
    section ? sectionForm(section, { base: "model.encoder.params" }) : null,
  ]), { meta: state.preview.resolved_config.model.encoder?.type || "identity" });
}

function encoderInspector() {
  return [head(icon("network"), "Encoder", "maps the dataset's input to the core"), sectionForm(state.schema.sections["model.encoder"], { base: "model.encoder", only: ["type"] }), sectionForm(state.schema.sections["model.encoder.params"], { base: "model.encoder.params" })];
}

function decoderInspector() {
  const section = state.schema.sections["model.decoder.params"];
  return [
    head(icon("sliders"), "Decoder", "reads the core's output and produces the prediction"),
    sectionForm(state.schema.sections["model.decoder"], { base: "model.decoder", only: ["type"] }),
    group("Shape", [sectionForm(section, { base: "model.decoder.params", only: ["hidden_dims", "activation", "output_dim"] })], state.preview.task === "regression" ? "Output dimension = number of predicted values." : "Output dimension = number of classes."),
    disclosure("decoder-all", "All decoder options", () => sectionForm(section, { base: "model.decoder.params", exclude: ["hidden_dims", "activation", "output_dim"] })),
  ];
}

function modelInspector() {
  const p = state.preview;
  const core = state.config.model.core;
  const types = state.schema.sections["model.core"].fields.find(f => f.name === "type")?.choices || [];
  const groups = {};
  for (const t of types) {
    const fam = state.schema.core_families?.[t];
    const label = fam === "population_network" ? "Population networks" : fam === "baseline_rnn" || fam === "heterogeneous_leak_ctrnn" || fam === "legendre_memory" ? "Recurrent baselines" : fam ? "Other payload families" : "Structured and feedforward cores";
    (groups[label] = groups[label] || []).push(t);
  }
  return [
    head(icon("layers"), familyLabel(p.family), p.recurrent ? "recurrent core" : "feedforward core"),
    group("Family", [
      field({ label: "Core type", control: selectInput(core.type, types, switchFamily, { groups, allowCustom: true }), key: "model.core.type", help: "Changing the family removes options the new family does not read; Undo restores them." }),
      sectionForm(state.schema.sections["model.core"], { base: "model.core", only: ["initialization_seed", "biological_neuron", "target_active_parameters"] }),
    ]),
    encoderGroup(),
    disclosure("decoder", "Decoder", () => h("div", {}, decoderInspector().slice(1)), { meta: decoderText().replace("Decoder · ", "") }),
  ];
}

// Structural edits ---------------------------------------------------------------------

async function switchFamily(type) {
  const core = state.config.model.core;
  if (type === core.type) return;
  const family = state.schema.core_families?.[type] ?? null;
  const removed = Object.keys(core).filter(k => k !== "type" && k !== "initialization_seed" && k !== "biological_neuron" && (family ? k !== family : PAYLOAD_KEYS.includes(k)));
  if (removed.length && !(await confirmDialog({ title: "Change the model family", text: `${familyLabel(type)} does not read ${removed.join(", ")}. These options will be removed (Undo restores them).`, confirm: "Change family" }))) return;
  await edit(cfg => {
    const c = cfg.model.core;
    for (const key of removed) delete c[key];
    c.type = type;
    if (family && !c[family]) c[family] = family === "population_network" ? starterNetwork() : {};
  }, { path: "model.core.type", structural: true });
  select(isPopulationNetwork() ? { kind: "network" } : { kind: "model" });
}

function starterNetwork() {
  const width = state.preview?.overview?.input_dim || 16;
  return {
    input_dim: width,
    layers: [
      {
        name: "layer0",
        population_defaults: { ff_excitatory_synapses: 8, ff_inhibitory_synapses: 4 },
        populations: [
          { name: "i", polarity: "inhibitory", n_neurons: 8, branch_factors: [1] },
          { name: "e", polarity: "excitatory", n_neurons: 16, branch_factors: [2, 2] },
        ],
        connections: [{ source: "input", target: "i" }, { source: "input", target: "e" }, { source: "i", target: "e" }],
      },
    ],
  };
}

function uniqueName(existing, stem) {
  let i = 1;
  let name = stem;
  while (existing.includes(name)) name = `${stem}${++i}`;
  return name;
}

function addPopulation(li, polarity) {
  const layer = layers()[li];
  const names = layer.populations.map(p => p.name);
  const name = uniqueName(names, polarity === "inhibitory" ? "i" : "e");
  const readout = state.preview.layers[li]?.readout;
  edit(cfg => {
    const l = getPath(cfg, `${NET}.layers.${li}`);
    l.populations.push({ name, polarity, n_neurons: 8, branch_factors: polarity === "inhibitory" ? [1] : [2] });
    l.connections = l.connections || [];
    l.connections.push({ source: "input", target: name });
    if (polarity === "inhibitory" && readout) l.connections.push({ source: name, target: readout });
  }, { structural: true }).then(() => select({ kind: "population", layer: li, population: names.length }));
}

function addConnection(li) {
  const layer = layers()[li];
  const target = state.preview.layers[li]?.readout || layer.populations[0]?.name;
  const index = (layer.connections || []).length;
  edit(cfg => {
    const l = getPath(cfg, `${NET}.layers.${li}`);
    l.connections = l.connections || [];
    l.connections.push({ source: "input", target });
  }, { structural: true }).then(() => select({ kind: "connection", layer: li, index }));
}

function addLayer() {
  const all = layers();
  const last = all.at(-1);
  // Read the new layer's index now: `all` is the live list the edit extends.
  const index = all.length;
  const name = uniqueName(all.map(l => l.name), `layer${index}`);
  edit(cfg => {
    const net = getPath(cfg, NET);
    net.layers.push({
      name,
      ...(last?.population_defaults ? { population_defaults: structuredClone(last.population_defaults) } : {}),
      populations: [
        { name: "i", polarity: "inhibitory", n_neurons: 8, branch_factors: [1] },
        { name: "e", polarity: "excitatory", n_neurons: 16, branch_factors: [2, 2] },
      ],
      connections: [{ source: "input", target: "i" }, { source: "input", target: "e" }, { source: "i", target: "e" }],
    });
    if (net.readout_layer === last?.name) delete net.readout_layer;
  }, { structural: true }).then(() => select({ kind: "layer", layer: index }));
}

async function removePopulation(li, pi) {
  const layer = layers()[li];
  const name = layer.populations[pi].name;
  const qualified = `${layer.name}.${name}`;
  const affected = (layer.connections || []).filter(c => c.source === name || c.target === name).length + layers().reduce((a, l) => a + (l.connections || []).filter(c => c.source === qualified).length, 0);
  if (!(await confirmDialog({ title: "Remove population", text: `Remove ${qualified}${affected ? ` and its ${describeCount(affected, "connection")}` : ""}? Undo restores it.`, confirm: "Remove" }))) return;
  await edit(cfg => {
    const net = getPath(cfg, NET);
    const l = net.layers[li];
    l.populations.splice(pi, 1);
    l.connections = (l.connections || []).filter(c => c.source !== name && c.target !== name);
    for (const other of net.layers) other.connections = (other.connections || []).filter(c => c.source !== qualified);
    if (l.readout_population === name) delete l.readout_population;
  }, { structural: true });
  select({ kind: "layer", layer: li });
}

async function removeConnection(li, index) {
  await edit(cfg => getPath(cfg, `${NET}.layers.${li}.connections`).splice(index, 1), { structural: true });
  const remaining = (layers()[li].connections || []).length;
  select(remaining ? { kind: "connection", layer: li, index: Math.min(index, remaining - 1) } : { kind: "layer", layer: li });
}

async function removeLayer(li) {
  const layer = layers()[li];
  if (!(await confirmDialog({ title: "Remove layer", text: `Remove ${layer.name} with its ${describeCount(layer.populations.length, "population")}? Undo restores it.`, confirm: "Remove" }))) return;
  await edit(cfg => {
    const net = getPath(cfg, NET);
    net.layers.splice(li, 1);
    const prefix = `${layer.name}.`;
    for (const other of net.layers) other.connections = (other.connections || []).filter(c => !String(c.source).startsWith(prefix));
    if (net.readout_layer === layer.name) delete net.readout_layer;
  }, { structural: true });
  select({ kind: "layer", layer: Math.max(0, li - 1) });
}

function renamePopulation(li, pi, name) {
  const layer = layers()[li];
  const old = layer.populations[pi].name;
  if (!name || name === old) return;
  if (!/^[A-Za-z_][\w-]*$/.test(name)) return toast("Use letters, digits, - and _ (starting with a letter).", { kind: "warn" });
  if (layer.populations.some(p => p.name === name)) return toast(`${layer.name} already has a population called ${name}.`, { kind: "warn" });
  edit(cfg => {
    const net = getPath(cfg, NET);
    const l = net.layers[li];
    l.populations[pi].name = name;
    for (const c of l.connections || []) {
      if (c.source === old) c.source = name;
      if (c.target === old) c.target = name;
    }
    if (l.readout_population === old) l.readout_population = name;
    for (const other of net.layers) for (const c of other.connections || []) if (c.source === `${l.name}.${old}`) c.source = `${l.name}.${name}`;
    if (net.readout_population === old && (net.readout_layer || net.layers.at(-1).name) === l.name) net.readout_population = name;
  }, { structural: true });
}

function renameLayer(li, name) {
  const old = layers()[li].name;
  if (!name || name === old) return;
  if (!/^[A-Za-z_][\w-]*$/.test(name)) return toast("Use letters, digits, - and _ (starting with a letter).", { kind: "warn" });
  if (layers().some(l => l.name === name)) return toast(`A layer called ${name} already exists.`, { kind: "warn" });
  edit(cfg => {
    const net = getPath(cfg, NET);
    net.layers[li].name = name;
    for (const other of net.layers) for (const c of other.connections || []) if (String(c.source).startsWith(`${old}.`)) c.source = `${name}.${String(c.source).slice(old.length + 1)}`;
    if (net.readout_layer === old) net.readout_layer = name;
  }, { structural: true });
}

function setFactors(li, pi, factors) {
  const itemPath = `${NET}.layers.${li}.populations.${pi}`;
  const item = layers()[li].populations[pi];
  const levels = factors.length + 1;
  const stale = STREAMS.map(([k]) => `${k}_synapses_by_level`).filter(k => Array.isArray(item.population?.[k]) && item.population[k].length && item.population[k].length !== levels);
  edit(cfg => {
    const it = getPath(cfg, itemPath);
    it.branch_factors = factors;
    for (const key of stale) delete it.population[key];
    if (it.population && !Object.keys(it.population).length) delete it.population;
  }, { path: `${itemPath}.branch_factors`, structural: true });
  if (stale.length) toast("Per-level contact counts of this population were reset because the number of levels changed.", { kind: "info" });
}

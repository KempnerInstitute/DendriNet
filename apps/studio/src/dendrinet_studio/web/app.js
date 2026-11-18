"use strict";
const $ = id => document.getElementById(id);
const externalInputs = new Set(["input", "input_e", "input_i"]);
const ns = "http://www.w3.org/2000/svg";
let token = location.hash.slice(1),
  config = null,
  preview = null,
  schema = null;
let selected = [0, 0],
  revision = 0,
  yamlDirty = false;

function node(tag, text, parent, attrs = {}) {
  const e = document.createElement(tag);
  if (text !== null) e.textContent = text;
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
  if (parent) parent.append(e);
  return e;
}

function svg(tag, attrs, parent, text) {
  const e = document.createElementNS(ns, tag);
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
  if (text !== undefined) e.textContent = text;
  parent.append(e);
  return e;
}

function message(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}
async function api(path, data) {
  const response = await fetch("/api/" + path, {
    method: data === undefined ? "GET" : "POST",
    headers: {
      "X-Studio-Token": token,
      "Content-Type": "application/json"
    },
    body: data === undefined ? undefined : JSON.stringify(data)
  });
  const result = await response.json();
  if (!response.ok) throw Error(result.error || response.statusText);
  return result;
}

function layers() {
  return config.model.core.population_network.layers;
}

function picked() {
  return layers()[selected[0]].populations[selected[1]];
}

function get(obj, path) {
  return path.split(".").reduce((o, k) => o?.[k], obj);
}

function set(obj, path, value) {
  const keys = path.split(".");
  let o = obj;
  keys.slice(0, -1).forEach(k => {
    o[k] ??= {};
    o = o[k];
  });
  o[keys.at(-1)] = value;
}

function button(label, parent, action, cls) {
  const b = node("button", label, parent, cls ? {
    class: cls
  } : {});
  b.onclick = () => Promise.resolve().then(action).catch(e => message(e.message, true));
  return b;
}

function field(parent, label, value, onChange, options = {}) {
  const wrap = node("label", label, parent);
  let input;
  if (options.choices) {
    input = node("select", null, wrap);
    const choices = [...options.choices];
    if (!choices.some(c => (Array.isArray(c) ? c[0] : c) === String(value))) choices.push(String(value));
    choices.forEach(c => {
      const [v, t] = Array.isArray(c) ? c : [c, c];
      node("option", t, input, {
        value: v
      });
    });
    input.value = String(value);
  } else {
    input = node("input", null, wrap, {
      type: typeof value === "boolean" ? "checkbox" : options.type || "text"
    });
    if (typeof value === "boolean") {
      input.checked = value;
      wrap.classList.add("check");
    } else input.value = value ?? "";
    if (options.type === "number") {
      input.step = options.step || "any";
      if (options.min !== undefined) input.min = options.min;
    }
  }
  input.setAttribute("aria-label", label);
  if (options.key) node("code", options.key, wrap);
  input.onchange = () => {
    try {
      input.setCustomValidity("");
      const v = input.type === "checkbox" ? input.checked : options.type === "number" ? Number(input.value) : input.value;
      if (options.type === "number" && (!Number.isFinite(v) || (options.integer && !Number.isInteger(v)))) throw Error(label + " must be a finite " + (options.integer ? "integer" : "number"));
      onChange(v);
      changed();
    } catch (e) {
      input.setCustomValidity(e.message);
      message(e.message, true);
    }
  };
  return input;
}
async function validate() {
  const invalid = document.querySelector("input:invalid");
  if (invalid) {
    message(invalid.validationMessage, true);
    invalid.reportValidity();
    return false;
  }
  const current = ++revision;
  try {
    const result = await api("preview", {
      config
    });
    if (current !== revision) return false;
    preview = result;
    if (!yamlDirty) $("yaml").value = result.yaml;
    $("metrics").replaceChildren();
    [
      [result.nodes.length, "populations"],
      [result.nodes.reduce((s, n) => s + n.neurons, 0), "neurons"],
      [result.compartments_total, "compartments incl. soma"]
    ].forEach(([v, label]) => {
      const m = node("div", null, $("metrics"), {
        class: "metric"
      });
      node("strong", v.toLocaleString(), m);
      node("span", label, m);
    });
    message(result.validation);
    return true;
  } catch (e) {
    if (current === revision) {
      preview = null;
      message(e.message, true);
    }
    return false;
  }
}
async function changed() {
  render();
  await validate();
}
async function adopt(result) {
  yamlDirty = false;
  config = result.config;
  preview = result;
  selected = [0, 0];
  await changed();
}

function renderLayers() {
  $("layers").replaceChildren();
  layers().forEach((layer, li) => {
    const group = node("div", null, $("layers"), {
      class: "layer-group"
    });
    node("strong", layer.name, group);
    layer.populations.forEach((p, pi) => button((p.polarity === "inhibitory" ? "○ I " : "△ E ") + p.name + " · " + p.n_neurons, group, () => {
      selected = [li, pi];
      render();
    }, "population-select" + (selected[0] === li && selected[1] === pi ? " active" : "")));
    for (const polarity of ["excitatory", "inhibitory"]) button("+ " + (polarity === "excitatory" ? "E" : "I"), group, () => {
      let name = (polarity === "excitatory" ? "e" : "i") + layer.populations.length;
      while (layer.populations.some(p => p.name === name)) name += "_";
      layer.populations.push({
        name,
        polarity,
        n_neurons: 8,
        branch_factors: [2],
        population: {
          ff_inhibitory_synapses: 0
        }
      });
      selected = [li, layer.populations.length - 1];
      return changed();
    }, "small");
  });
}

function renderSettings() {
  const parent = $("population");
  parent.replaceChildren();
  const [li, pi] = selected, layer = layers()[li], p = picked();
  const info = preview?.nodes.find(n => n.layer === li && n.population === pi);
  const setting = k => p.population?.[k] ?? layer.population_defaults?.[k] ?? info?.settings[k] ?? schema.population_fields[k]?.default;
  const update = (k, v) => {
    p.population ??= {};
    p.population[k] = v;
  };
  field(parent, "Population name", p.name, value => {
    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(value)) throw Error("Use letters, digits and underscores for names");
    const old = p.name;
    layers().forEach((l, index) => l.connections.forEach(c => {
      if (c.source === layer.name + "." + old) c.source = layer.name + "." + value;
      if (index === li) {
        if (c.source === old) c.source = value;
        if (c.target === old) c.target = value;
      }
    }));
    if (layer.readout_population === old) layer.readout_population = value;
    p.name = value;
  });
  field(parent, "Neuron count", p.n_neurons, v => p.n_neurons = v, {
    type: "number",
    min: 1,
    integer: true,
    key: "n_neurons"
  });
  field(parent, "Branching factors (empty = soma only)", p.branch_factors.join(", "), v => {
    const parts = v.replace(/[\[\]]/g, "").trim();
    const factors = parts ? parts.split(/[ ,]+/).map(Number) : [];
    if (factors.some(x => !Number.isInteger(x) || x < 1)) throw Error("Branching factors must be positive integers");
    p.branch_factors = factors;
  }, {
    key: "branch_factors"
  });
  field(parent, "Integration", setting("use_shunting") ? "shunting" : "additive", v => update("use_shunting", v === "shunting"), {
    choices: ["shunting", "additive"],
    key: "population.use_shunting"
  });
  const numeric = [
    ["E contacts per compartment", "ff_excitatory_synapses"],
    ["I contacts per compartment", "ff_inhibitory_synapses"],
    ["Leak conductance", "leak_conductance"]
  ];
  numeric.forEach(([label, k]) => field(parent, label, setting(k), v => update(k, v), {
    type: "number",
    min: k === "leak_conductance" ? 0.000001 : 0,
    integer: k !== "leak_conductance",
    key: "population." + k
  }));
  [
    ["Somatic synapses", "somatic_synapses"],
    ["Learn leak", "learnable_leak"],
    ["Reactivation gate", "reactivate"]
  ].forEach(([label, k]) => field(parent, label, setting(k), v => update(k, v), {
    key: "population." + k
  }));
  field(parent, "Sparse connections", setting("topk_type"), v => update("topk_type", v), {
    choices: [
      ["indexed", "Fixed indexed"],
      ["standard", "Learned Top-K"],
      ["indexed_rewire", "Indexed rewiring"]
    ],
    key: "population.topk_type"
  });
  field(parent, "Weight transform", setting("weight_transform"), v => update("weight_transform", v), {
    choices: ["softplus", "exp", "identity", "relu"],
    key: "population.weight_transform"
  });
  field(parent, "Layer readout", layer.readout_population || "", v => layer.readout_population = v || null, {
    choices: layer.populations.map(p => p.name),
    key: "readout_population"
  });
  button("Remove selected population", parent, () => {
    if (layer.populations.length === 1) throw Error("A layer must contain a population");
    const id = layer.name + "." + p.name;
    layers().forEach((l, index) => {
      l.connections = l.connections.filter(c => c.source !== id && !(index === li && (c.source === p.name || c.target === p.name)));
    });
    layer.populations.splice(pi, 1);
    if (layer.readout_population === p.name) layer.readout_population = layer.populations.at(-1).name;
    selected = [li, 0];
    return changed();
  }, "small");
}

function renderConnections() {
  const parent = $("connections");
  parent.replaceChildren();
  const li = selected[0],
    layer = layers()[li];
  const sources = ["input", "input_e", "input_i", ...layer.populations.map(p => p.name), ...layers().slice(0, li).flatMap(l => l.populations.map(p => l.name + "." + p.name))];
  layer.connections.forEach((c, index) => {
    const box = node("details", null, parent, {
      class: "connection"
    });
    node("summary", c.source + " → " + c.target, box);
    field(box, "Source", c.source, v => c.source = v, {
      choices: sources
    });
    field(box, "Target", c.target, v => c.target = v, {
      choices: layer.populations.map(p => p.name)
    });
    field(box, "Target levels (blank = all)", (c.target_levels || []).join(", "), v => {
      c.target_levels = v.trim() ? v.split(/[ ,]+/).map(x => /^\d+$/.test(x) ? Number(x) : x) : null;
    }, {
      key: "target_levels"
    });
    field(box, "Timing", c.timing || "same_step", v => c.timing = v, {
      choices: ["same_step", "delayed"]
    });
    field(box, "Enabled", c.enabled ?? true, v => c.enabled = v);
    button("Remove", box, () => {
      layer.connections.splice(index, 1);
      return changed();
    }, "small");
  });
}

function renderExperiment() {
  const parent = $("experiment");
  parent.replaceChildren();
  node("p", "Dataset: " + config.data.dataset_name, parent);
  [
    ["Seed", "experiment.seed", 42, true],
    ["Epochs", "training.main.common.epochs", 3, true],
    ["Batch size", "training.main.common.batch_size", 64, true],
    ["Base learning rate", "training.main.common.param_groups.lr", .001, false],
    ["Synaptic decay", "training.main.common.synapse_decay_rate", 0, false],
    ["Adam weight decay", "training.main.optimizer.weight_decay", 0, false]
  ].forEach(([label, path, fallback, integer]) => field(parent, label, get(config, path) ?? get(preview?.resolved_config, path) ?? fallback, value => set(config, path, value), {
    type: "number",
    integer,
    key: path
  }));
  field(parent, "Weight boosting", get(config, "training.main.common.weight_boosting") ?? false, v => set(config, "training.main.common.weight_boosting", v), {
    key: "training.main.common.weight_boosting"
  });
  node("p", "Component learning rates, initialization and other advanced settings remain in YAML. Changing the base rate does not override explicit component rates.", parent, {
    class: "hint"
  });
  $("biological").checked = config.model.core.biological_neuron ?? true;
}

function renderGraph() {
  const canvas = $("graph");
  canvas.replaceChildren();
  const height = Math.max(370, 85 + Math.max(...layers().map(l => l.populations.length)) * 135);
  const width = Math.max(480, 190 + layers().length * 260);
  canvas.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const defs = svg("defs", {}, canvas);
  for (const [id, color, d] of [
      ["arrow", "#426a80", "M0,0 L8,4 L0,8 Z"],
      ["bar", "#a15000", "M6,0 L6,8"]
    ]) {
    const marker = svg("marker", {
      id,
      viewBox: "0 0 8 8",
      refX: 7,
      refY: 4,
      markerWidth: 8,
      markerHeight: 8,
      orient: "auto"
    }, defs);
    svg("path", {
      d,
      fill: id === "arrow" ? color : "none",
      stroke: color,
      "stroke-width": 2
    }, marker);
  }
  const positions = {
    input: {
      x: 70,
      y: height / 2,
      polarity: "excitatory"
    },
    input_e: {
      x: 70,
      y: height / 2 - 65,
      polarity: "excitatory"
    },
    input_i: {
      x: 70,
      y: height / 2 + 65,
      polarity: "inhibitory"
    }
  };
  layers().forEach((l, li) => l.populations.forEach((p, pi) => {
    positions[l.name + "." + p.name] = {
      x: 260 + li * 260,
      y: 100 + pi * 135,
      polarity: p.polarity
    };
  }));
  const external = new Set();
  layers().forEach((l) => l.connections.forEach(c => {
    const sourceId = c.source.includes(".") || externalInputs.has(c.source) ? c.source : l.name + "." + c.source;
    const from = positions[sourceId],
      to = positions[l.name + "." + c.target];
    if (externalInputs.has(sourceId)) external.add(sourceId);
    if (!from || !to) return;
    const inhibitory = c.pathway?.includes("inhibitory") || from.polarity === "inhibitory";
    const d = from.x === to.x ? `M${from.x + 78},${from.y} C${from.x + 140},${from.y} ${to.x + 140},${to.y} ${to.x + 78},${to.y}` : `M${from.x + 65},${from.y} C${from.x + 130},${from.y} ${to.x - 125},${to.y} ${to.x - 78},${to.y}`;
    const path = svg("path", {
      d,
      class: "edge" + (inhibitory ? " inhibitory" : ""),
      "marker-end": inhibitory ? "url(#bar)" : "url(#arrow)",
      "stroke-dasharray": c.enabled === false ? "2 5" : c.timing === "delayed" ? "7 4" : "none",
      opacity: c.enabled === false ? .4 : 1
    }, canvas);
    svg("title", {}, path, `${c.source} → ${c.target}; ${c.timing || "same_step"}; targets ${(c.target_levels || ["all levels"]).join(", ")}${c.enabled === false ? "; disabled" : ""}`);
    if (c.timing === "delayed" || c.enabled === false) svg("text", {
      x: (from.x + to.x) / 2 + 90,
      y: (from.y + to.y) / 2
    }, canvas, c.enabled === false ? "disabled" : "delayed");
  }));
  external.forEach(id => {
    const p = positions[id];
    svg("rect", {
      x: p.x - 45,
      y: p.y - 18,
      width: 90,
      height: 36,
      rx: 18,
      fill: "#e2e9ee",
      stroke: "#819aa9"
    }, canvas);
    svg("text", {
      x: p.x,
      y: p.y + 4,
      "text-anchor": "middle"
    }, canvas, id);
  });
  layers().forEach((l, li) => {
    svg("text", {
      x: 260 + li * 260,
      y: 28,
      "text-anchor": "middle",
      "font-weight": 650
    }, canvas, l.name);
    l.populations.forEach((p, pi) => {
      const pos = positions[l.name + "." + p.name];
      const g = svg("g", {
        class: "node" + (li === selected[0] && pi === selected[1] ? " selected" : ""),
        tabindex: 0,
        role: "button",
        "aria-label": `Edit ${l.name}.${p.name}`,
        transform: `translate(${pos.x - 78},${pos.y - 42})`
      }, canvas);
      const action = () => {
        selected = [li, pi];
        render();
      };
      g.onclick = action;
      g.onkeydown = e => {
        if (["Enter", " "].includes(e.key)) {
          e.preventDefault();
          action();
        }
      };
      svg("rect", {
        x: 0,
        y: 0,
        width: 156,
        height: 84,
        rx: 10,
        class: "card"
      }, g);
      if (p.polarity === "inhibitory") svg("circle", {
        cx: 21,
        cy: 25,
        r: 9,
        fill: "#fff",
        stroke: "#a15000",
        "stroke-width": 2
      }, g);
      else svg("path", {
        d: "M12,33 L21,15 L30,33 Z",
        fill: "#fff",
        stroke: "#006b87",
        "stroke-width": 2
      }, g);
      svg("text", {
        x: 38,
        y: 30,
        class: "name"
      }, g, (p.polarity === "inhibitory" ? "I · " : "E · ") + p.name);
      svg("text", {
        x: 16,
        y: 53
      }, g, p.n_neurons + " neurons");
      svg("text", {
        x: 16,
        y: 71
      }, g, p.branch_factors.length ? "[" + p.branch_factors.join(", ") + "]" : "Soma only");
    });
  });
}

function renderTree() {
  const tree = $("tree");
  tree.replaceChildren();
  const p = picked();
  $("treeTitle").textContent = "Neuron morphology · " + p.name;
  let widths = [1];
  p.branch_factors.forEach(n => widths.push(widths.at(-1) * n));
  const count = widths.reduce((a, b) => a + b, 0);
  $("treeCount").textContent = count.toLocaleString() + " compartments including soma";
  if (count > 1500) {
    $("treeNote").textContent = "This morphology has " + count.toLocaleString() + " compartments per neuron. Individual-compartment drawing is limited to 1,500; all counts and configuration values are retained.";
    return;
  }
  $("treeNote").textContent = "Exact branching-factor tree; illustrative geometry, not cable lengths or sampled synapses. Soma shown as a triangle.";
  const levels = widths.map((n, level) => Array.from({
    length: n
  }, (_, i) => ({
    x: (i + .5) * 860 / n + 20,
    y: 235 - level * 195 / Math.max(1, widths.length - 1)
  })));
  levels.slice(1).forEach((points, index) => points.forEach((point, child) => {
    const parent = levels[index][Math.floor(child / p.branch_factors[index])];
    svg("line", {
      x1: parent.x,
      y1: parent.y,
      x2: point.x,
      y2: point.y,
      stroke: "#547482",
      "stroke-width": 1.7
    }, tree);
  }));
  levels.forEach((points, index) => points.forEach(pt => index ? svg("circle", {
    cx: pt.x,
    cy: pt.y,
    r: Math.min(5, 370 / widths[index]),
    fill: "#fff",
    stroke: "#294e62",
    "stroke-width": 1.3
  }, tree) : svg("path", {
    d: `M${pt.x-10},${pt.y+9} L${pt.x},${pt.y-10} L${pt.x+10},${pt.y+9} Z`,
    fill: "#006b87"
  }, tree)));
}

function render() {
  if (!config) return;
  renderLayers();
  renderSettings();
  renderConnections();
  renderExperiment();
  renderGraph();
  renderTree();
}

function switchTab(id) {
  document.querySelectorAll(".view").forEach(e => e.classList.toggle("hidden", e.id !== id));
  document.querySelectorAll(".tab").forEach(e => e.classList.toggle("active", e.dataset.tab === id));
  if (id === "runs") refreshRuns();
}

function lossPlot(parent, summary) {
  const train = summary.train_losses,
    valid = summary.valid_losses;
  if (!Array.isArray(train) || !train.length) return;
  const chart = node("div", null, parent);
  node("p", "Loss by epoch: solid circles = training; dashed squares = validation", chart, {
    class: "curve-key"
  });
  const plot = document.createElementNS(ns, "svg");
  plot.setAttribute("viewBox", "0 0 650 220");
  chart.append(plot);
  const all = [...train, ...(valid || [])].filter(Number.isFinite),
    lo = Math.min(...all),
    hi = Math.max(...all),
    span = Math.max(1e-8, hi - lo);
  svg("path", {
    d: "M45,10 V180 H630",
    fill: "none",
    stroke: "#7893a3"
  }, plot);
  [train, valid || []].forEach((values, kind) => {
    const pts = values.map((v, i) => [45 + i * 575 / Math.max(1, train.length - 1), 170 - (v - lo) * 145 / span]);
    svg("polyline", {
      points: pts.map(p => p.join(",")).join(" "),
      fill: "none",
      stroke: kind ? "#a15000" : "#006b87",
      "stroke-width": 2,
      "stroke-dasharray": kind ? "7 4" : "none"
    }, plot);
    pts.forEach(([x, y]) => kind ? svg("rect", {
      x: x - 3,
      y: y - 3,
      width: 6,
      height: 6,
      fill: "#fff",
      stroke: "#a15000"
    }, plot) : svg("circle", {
      cx: x,
      cy: y,
      r: 3,
      fill: "#006b87"
    }, plot));
  });
  svg("text", {
    x: 6,
    y: 30
  }, plot, hi.toPrecision(3));
  svg("text", {
    x: 6,
    y: 174
  }, plot, lo.toPrecision(3));
  svg("text", {
    x: 43,
    y: 202
  }, plot, "1");
  svg("text", {
    x: 610,
    y: 202
  }, plot, String(train.length));
  svg("text", {
    x: 310,
    y: 213
  }, plot, "Epoch");
}
async function refreshRuns() {
  try {
    const rows = await api("runs"),
      parent = $("runList");
    parent.replaceChildren();
    if (!rows.length) node("p", "No runs yet. Check a model or train from the Design tab.", parent);
    rows.forEach(row => {
      const box = node("article", null, parent, {
        class: "run"
      });
      node("h2", row.kind + " · " + row.status, box);
      node("p", row.id, box);
      node("p", "Core " + row.core_version + " · " + row.device, box);
      if (row.status === "running") button("Stop this worker", box, async () => {
        await api("stop", {
          id: row.id
        });
        refreshRuns();
      });
      if (row.resources) {
        const table = node("table", null, box);
        ["total_parameters", "trainable_parameters", "active_synapses", "candidate_synapse_slots"].forEach(k => {
          const tr = node("tr", null, table);
          node("th", k.replaceAll("_", " "), tr);
          node("td", row.resources[k]?.toLocaleString() || "—", tr);
        });
        node("p", row.resources.scope, box, {
          class: "hint"
        });
      }
      if (row.summary) {
        lossPlot(box, row.summary);
        const d = node("details", null, box);
        node("summary", "Training summary", d);
        node("pre", JSON.stringify(row.summary, null, 2), d);
      }
      const d = node("details", null, box, row.status === "failed" ? {
        open: ""
      } : {});
      node("summary", "Worker log and command", d);
      node("pre", row.command.join(" ") + "\n\n" + (row.log || "Starting…"), d);
      node("p", "Saved in " + row.cwd, box, {
        class: "hint"
      });
    });
  } catch (e) {
    message(e.message, true);
  }
}
document.querySelectorAll(".tab").forEach(b => b.onclick = () => switchTab(b.dataset.tab));
$("loadExample").onclick = () => api("example/" + $("example").value).then(adopt).catch(e => message(e.message, true));
$("biological").onchange = () => {
  config.model.core.biological_neuron = $("biological").checked;
  changed();
};
$("addLayer").onclick = () => {
  const previous = layers().at(-1),
    l = structuredClone(previous);
  l.name = "layer" + layers().length;
  while (layers().some(x => x.name === l.name)) l.name += "_";
  const source = previous.name + "." + (previous.readout_population || previous.populations.at(-1).name);
  l.connections = l.connections.map(c => ({
    ...c,
    source: externalInputs.has(c.source) || c.source.includes(".") ? source : c.source
  }));
  layers().push(l);
  selected = [layers().length - 1, 0];
  changed();
};
$("addConnection").onclick = () => {
  const l = layers()[selected[0]];
  l.connections.push({
    source: "input",
    target: picked().name
  });
  changed();
};
$("upload").onchange = async e => {
  try {
    await adopt(await api("preview", {
      yaml: await e.target.files[0].text()
    }));
  } catch (error) {
    message(error.message, true);
  }
};
$("yaml").oninput = () => {
  yamlDirty = true;
  message("YAML edits have not been applied. Apply YAML before exporting or running.");
};
$("applyYaml").onclick = async () => {
  try {
    await adopt(await api("preview", {
      yaml: $("yaml").value
    }));
  } catch (e) {
    message(e.message, true);
  }
};
$("export").onclick = async () => {
  if (yamlDirty) {
    message("Apply YAML before exporting.", true);
    return;
  }
  if (!await validate()) return;
  const url = URL.createObjectURL(new Blob([preview.yaml], {
    type: "text/yaml"
  }));
  const a = node("a", null, null, {
    href: url,
    download: "dendrinet-config.yaml"
  });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
for (const [id, kind] of [
    ["inspect", "inspect"],
    ["train", "train"]
  ]) $(id).onclick = async () => {
  try {
    if (yamlDirty) throw Error("Apply YAML before running.");
    if (!await validate()) return;
    await api("run", {
      config,
      kind,
      device: $("device").value
    });
    switchTab("runs");
    message("Worker started. The saved config and run record identify this execution.");
  } catch (e) {
    message(e.message, true);
  }
};
$("refreshRuns").onclick = refreshRuns;
setInterval(() => {
  if (!$("runs").classList.contains("hidden")) refreshRuns();
}, 5000);
(async () => {
  try {
    schema = await api("schema");
    $("version").textContent = "Core " + schema.core_version;
    schema.examples.forEach(name => node("option", name.replaceAll("_", " "), $("example"), {
      value: name
    }));
    $("example").value = "synthetic_population";
    await adopt(await api("example/synthetic_population"));
  } catch (e) {
    message(e.message, true);
  }
})();

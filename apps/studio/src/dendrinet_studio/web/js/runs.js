// The Runs view: every model check and training run in the workspace, with
// overlaid curves, final metrics, configuration differences and run details.

import { h, icon, fmt, pct, duration, timeOfDay, download, copyText } from "./dom.js";
import { state, api, emit, adopt } from "./store.js";
import { lineChart, dotChart, runSwatch, curvesCsv, humanMetric, isFractionMetric, SLOTS } from "./charts.js";
import { toast, promptDialog } from "./ui.js";

const slots = new Map();
const configs = new Map();
let kindFilter = "train";
let metricChoice = null;
let lastCompareKey = "";
let lastDetailKey = "";
let root = null;

const STATUS = {
  running: ["dot", "Running", "running"],
  completed: ["check", "Completed", "ok"],
  failed: ["cross", "Failed", "error"],
  "failed to start": ["cross", "Failed to start", "error"],
  stopped: ["stop", "Stopped", "muted"],
};

export function runTitle(run) {
  if (run.name) return run.name;
  const what = run.kind === "inspect" ? "Model check" : "Training";
  return `${what} · ${run.dataset || run.family || "run"} · ${timeOfDay(run.started)}`;
}

export function statusBadge(run) {
  const known = STATUS[run.status];
  const [ic, text, cls] = known || ["warn", run.status.charAt(0).toUpperCase() + run.status.slice(1), "warn"];
  let label = text;
  if (run.status === "failed" && run.exit_code != null) label = `${text} (exit ${run.exit_code})`;
  if (run.status === "running") {
    const p = run.progress;
    label = p ? `${text} · epoch ${p.epoch} of ${p.epochs}` : `${text} · ${duration(run.duration_s)}`;
  }
  return h("span", { class: `status ${cls}` }, [icon(ic), label]);
}

function assignSlot(id) {
  if (slots.has(id)) return slots.get(id);
  const used = new Set(slots.values());
  for (let i = 0; i < SLOTS; i++) {
    if (!used.has(i)) {
      slots.set(id, i);
      return i;
    }
  }
  return null;
}

function primaryMetric(run) {
  const m = run.metrics || {};
  for (const name of ["accuracy", "r2", "mse", "auc"]) if (m[name]) return name;
  return Object.keys(m)[0] || null;
}

function formatMetric(name, value) {
  if (!Number.isFinite(value)) return "—";
  return isFractionMetric(name) ? pct(value) : fmt(value, 4);
}

function headline(run) {
  const parts = [];
  if (run.kind === "inspect") {
    if (run.resources) parts.push(`${fmt(run.resources.trainable_parameters)} parameters`, `${fmt(run.resources.active_synapses)} active synapses`);
  } else {
    const name = primaryMetric(run);
    if (name) {
      const values = run.metrics[name];
      const label = name === "accuracy" ? "" : `${humanMetric(name)} `;
      if (Number.isFinite(values.valid)) parts.push(`val ${label}${formatMetric(name, values.valid)}`);
      if (Number.isFinite(values.test)) parts.push(`test ${formatMetric(name, values.test)}`);
    } else if (run.summary && Number.isFinite(run.summary.best_loss)) {
      parts.push(`best val loss ${fmt(run.summary.best_loss, 4)}`);
    }
  }
  if (run.status !== "running" && run.duration_s != null) parts.push(duration(run.duration_s));
  parts.push(run.device === "auto" ? "GPU if available" : "CPU");
  return parts.join(" · ");
}

// Layout ----------------------------------------------------------------------

export function mountRuns(container) {
  root = container;
  root.replaceChildren(
    h("div", { class: "runs-layout" }, [
      h("aside", { class: "run-list", "aria-label": "Runs" }, [
        h("div", { class: "run-list-head" }, [
          h("h2", { class: "eyebrow", text: "Runs in this workspace" }),
          h("div", { class: "segmented small", role: "radiogroup", "aria-label": "Show", id: "run-filter" }),
        ]),
        h("div", { id: "run-problems" }),
        h("div", { id: "run-cards", class: "run-cards" }),
        h("p", { class: "help", id: "run-hint", text: "Tick two or more training runs to overlay them. Each run keeps its YAML, command, package version and log in its own folder." }),
      ]),
      h("section", { class: "run-main" }, [
        h("div", { class: "panel" }, [
          h("div", { class: "panel-head" }, [h("h3", { text: "Loss by epoch" }), h("span", { class: "sub", text: "solid = training · dashed and open markers = validation" }), h("span", { class: "spacer" }), h("button", { type: "button", class: "btn small", id: "export-curves", onclick: exportCurves }, [icon("download"), "Curves (CSV)"])]),
          h("div", { class: "panel-body", id: "loss-chart" }),
        ]),
        h("div", { class: "panel" }, [
          h("div", { class: "panel-head" }, [h("h3", { text: "Final results" }), h("span", { class: "sub", text: "after training, on each split" }), h("span", { class: "spacer" }), h("label", { class: "inline-label", for: "metric-choice", text: "Measure" }), h("select", { id: "metric-choice", class: "input compact" })]),
          h("div", { class: "panel-body", id: "metric-chart" }),
          h("div", { class: "panel-body table-scroll", id: "metric-table" }),
        ]),
        h("div", { class: "panel" }, [
          h("div", { class: "panel-head" }, [h("h3", { text: "What differs between the selected runs" }), h("span", { class: "sub", text: "from the saved configurations" })]),
          h("div", { class: "panel-body table-scroll", id: "run-diff" }),
        ]),
        h("div", { class: "panel", id: "run-detail" }),
      ]),
    ])
  );
  root.querySelector("#metric-choice").addEventListener("change", event => {
    metricChoice = event.target.value;
    lastCompareKey = "";
    updateRuns();
  });
  const filter = root.querySelector("#run-filter");
  for (const [value, label] of [["train", "Training"], ["inspect", "Model checks"], ["all", "All"]]) {
    filter.append(h("button", { type: "button", role: "radio", "aria-checked": String(kindFilter === value), text: label, onclick: () => {
      kindFilter = value;
      for (const b of filter.children) b.setAttribute("aria-checked", String(b.textContent === label));
      updateRuns();
    } }));
  }
  updateRuns();
}

export function updateRuns() {
  if (!root || !state.runs) return;
  const runs = state.runs.runs;
  // Drop selections of runs that no longer exist.
  for (const id of [...state.selectedRuns]) if (!runs.some(r => r.id === id)) state.selectedRuns.delete(id);
  for (const id of [...slots.keys()]) if (!state.selectedRuns.has(id)) slots.delete(id);
  if (!state.focusRun && runs.length) state.focusRun = runs[0].id;
  renderProblems();
  renderCards(runs);
  renderCompare(runs);
  renderDetail(runs.find(r => r.id === state.focusRun));
}

function renderProblems() {
  const box = root.querySelector("#run-problems");
  const problems = state.runs.problems || [];
  box.replaceChildren(
    ...problems.map(p => h("div", { class: "notice warn" }, [icon("warn"), h("span", { text: `${p.error}: ${p.path}` })]))
  );
}

function renderCards(runs) {
  const list = root.querySelector("#run-cards");
  const shown = runs.filter(r => kindFilter === "all" || r.kind === kindFilter);
  if (!shown.length) {
    list.replaceChildren(h("div", { class: "empty" }, [
      h("p", { text: kindFilter === "inspect" ? "No model checks yet." : "No training runs yet." }),
      h("p", { class: "help", text: state.runs.execution === false ? "This Studio runs in design-only mode: export the YAML and run it with dendritic-train." : "Use Check model or Train at the top right. Runs are saved in the workspace folder below." }),
      h("code", { class: "path", text: state.runs.workspace }),
    ]));
    return;
  }
  list.replaceChildren(
    ...shown.map(run => {
      const selected = state.selectedRuns.has(run.id);
      const comparable = run.kind === "train";
      const slot = selected ? slots.get(run.id) : null;
      const check = h("input", { type: "checkbox", "aria-label": `Compare ${runTitle(run)}`, disabled: !comparable || (!selected && state.selectedRuns.size >= SLOTS) });
      check.checked = selected;
      check.addEventListener("change", () => {
        if (check.checked) {
          if (assignSlot(run.id) === null) {
            check.checked = false;
            toast(`At most ${SLOTS} runs can be overlaid at once.`, { kind: "warn" });
            return;
          }
          state.selectedRuns.add(run.id);
        } else {
          state.selectedRuns.delete(run.id);
          slots.delete(run.id);
        }
        lastCompareKey = "";
        updateRuns();
      });
      const card = h("article", { class: "run-card" + (state.focusRun === run.id ? " focused" : "") + (selected ? " selected" : "") }, [
        h("div", { class: "run-card-top" }, [
          comparable ? check : h("span", { class: "check-spacer" }),
          h("button", { type: "button", class: "run-title", text: runTitle(run), onclick: () => {
            state.focusRun = run.id;
            lastDetailKey = "";
            updateRuns();
          } }),
          selected && slot !== null ? runSwatch(slot) : null,
        ]),
        h("div", { class: "run-card-meta" }, [statusBadge(run), run.kind === "inspect" ? h("span", { class: "tag", text: "model check" }) : null]),
        run.status === "running" && run.progress ? h("progress", { max: run.progress.epochs, value: run.progress.epoch, "aria-label": "Training progress" }) : null,
        h("p", { class: "run-card-line", text: headline(run) }),
      ]);
      return card;
    })
  );
}

function compareRuns(runs) {
  return runs
    .filter(r => state.selectedRuns.has(r.id))
    .map(r => ({
      id: r.id,
      label: runTitle(r),
      slot: slots.get(r.id) ?? assignSlot(r.id),
      train: r.summary?.train_losses || liveCurve(r, "train"),
      valid: r.summary?.valid_losses || liveCurve(r, "valid"),
      run: r,
    }));
}

function liveCurve(run, split) {
  const points = run.progress?.points;
  if (!points?.length) return [];
  const values = [];
  for (const p of points) values[p.epoch - 1] = split === "train" ? p.train_loss : p.valid_loss;
  return Array.from(values, v => (v === undefined ? NaN : v));
}

function renderCompare(runs) {
  let chosen = compareRuns(runs);
  // Nothing ticked: show the focused training run on its own.
  if (!chosen.length) {
    const focus = runs.find(r => r.id === state.focusRun && r.kind === "train");
    if (focus) chosen = [{ id: focus.id, label: runTitle(focus), slot: 0, train: focus.summary?.train_losses || liveCurve(focus, "train"), valid: focus.summary?.valid_losses || liveCurve(focus, "valid"), run: focus }];
  }
  const key = JSON.stringify(chosen.map(c => [c.id, c.slot, c.label, c.train.length, c.valid.length, c.run.status, c.run.metrics])) + metricChoice;
  if (key === lastCompareKey) return;
  lastCompareKey = key;
  lineChart(root.querySelector("#loss-chart"), chosen);
  root.querySelector("#export-curves").disabled = !chosen.some(c => c.train.length || c.valid.length);
  // Metrics.
  const names = [...new Set(chosen.flatMap(c => Object.keys(c.run.metrics || {})))];
  const select = root.querySelector("#metric-choice");
  if (!names.includes(metricChoice)) metricChoice = names.includes("accuracy") ? "accuracy" : names[0] || null;
  select.replaceChildren(...names.map(n => h("option", { value: n, text: humanMetric(n) })));
  select.value = metricChoice || "";
  select.disabled = !names.length;
  const rows = chosen.map(c => ({
    id: c.id,
    label: c.label,
    sub: [c.run.dataset, c.run.family].filter(Boolean).join(" · "),
    slot: c.slot,
    values: (c.run.metrics || {})[metricChoice] || {},
  }));
  const classes = chosen.map(c => c.run.outputs).filter(Boolean);
  const chance = metricChoice === "accuracy" && classes.length === chosen.length && new Set(classes).size === 1 && chosen.every(c => c.run.task === "classification") ? 1 / classes[0] : null;
  dotChart(root.querySelector("#metric-chart"), metricChoice ? rows : [], { metric: metricChoice || "metric", fraction: isFractionMetric(metricChoice), chance });
  root.querySelector("#metric-table").replaceChildren(metricTable(chosen));
  renderDiff(chosen);
}

function metricTable(chosen) {
  const names = [...new Set(chosen.flatMap(c => Object.keys(c.run.metrics || {})))];
  if (!names.length) return h("span");
  const table = h("table", { class: "data" });
  table.append(h("thead", {}, [h("tr", {}, [h("th", { text: "Run" }), ...names.flatMap(n => ["training", "validation", "test"].map(s => h("th", { text: `${humanMetric(n)} · ${s}` })))])]));
  const body = h("tbody");
  for (const c of chosen) {
    const row = h("tr", {}, [h("th", { scope: "row", class: "run-cell" }, [runSwatch(c.slot, { size: 12 }), " ", c.label])]);
    for (const n of names) for (const split of ["train", "valid", "test"]) row.append(h("td", { class: "num", text: formatMetric(n, (c.run.metrics || {})[n]?.[split]) }));
    body.append(row);
  }
  table.append(body);
  return table;
}

// Configuration differences ------------------------------------------------------

function flatten(value, prefix = "", out = {}) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const keys = Object.keys(value);
    if (!keys.length && prefix) out[prefix] = "{}";
    for (const k of keys) flatten(value[k], prefix ? `${prefix}.${k}` : k, out);
  } else if (Array.isArray(value) && value.some(v => v && typeof v === "object")) {
    value.forEach((v, i) => flatten(v, `${prefix}.${i}`, out));
  } else {
    out[prefix] = JSON.stringify(value);
  }
  return out;
}

async function loadConfig(id) {
  if (configs.has(id)) return configs.get(id);
  const result = await api(`runs/${encodeURIComponent(id)}/config`);
  configs.set(id, result.config);
  return result.config;
}

async function renderDiff(chosen) {
  const box = root.querySelector("#run-diff");
  if (chosen.length < 2) {
    box.replaceChildren(h("p", { class: "empty", text: "Select two or more runs to see which settings differ." }));
    return;
  }
  box.replaceChildren(h("p", { class: "muted", text: "Reading the saved configurations…" }));
  let flat;
  try {
    flat = await Promise.all(chosen.map(async c => flatten(await loadConfig(c.id))));
  } catch (error) {
    box.replaceChildren(h("div", { class: "notice error" }, [icon("cross"), h("span", { text: error.message })]));
    return;
  }
  const paths = [...new Set(flat.flatMap(f => Object.keys(f)))].sort();
  const differing = paths.filter(p => new Set(flat.map(f => f[p] ?? "—")).size > 1 && !p.startsWith("outputs."));
  if (!differing.length) {
    box.replaceChildren(h("p", { class: "muted", text: `All ${fmt(paths.length)} settings are identical (apart from output names).` }));
    return;
  }
  const table = h("table", { class: "data diff" });
  table.append(h("thead", {}, [h("tr", {}, [h("th", { text: "Setting" }), ...chosen.map(c => h("th", {}, [runSwatch(c.slot, { size: 12 }), " ", c.label]))])]));
  const body = h("tbody");
  for (const p of differing.slice(0, 80)) {
    body.append(h("tr", {}, [h("th", { scope: "row" }, [h("code", { text: p })]), ...flat.map(f => h("td", { class: "mono", text: f[p] === undefined ? "(not set)" : f[p] }))]));
  }
  table.append(body);
  const extra = differing.length > 80 ? h("p", { class: "help", text: `${differing.length - 80} more differences are not shown.` }) : null;
  box.replaceChildren(table, h("p", { class: "help", text: `${fmt(paths.length - differing.length)} other settings are identical.` }), extra);
}

// Details ---------------------------------------------------------------------------

function renderDetail(run) {
  const panel = root.querySelector("#run-detail");
  if (!run) {
    panel.replaceChildren(h("div", { class: "panel-head" }, [h("h3", { text: "Run details" })]), h("div", { class: "panel-body" }, [h("p", { class: "empty", text: "Select a run to see its details." })]));
    lastDetailKey = "";
    return;
  }
  const key = JSON.stringify([run.id, run.status, run.name, run.notes, run.metrics, run.resources?.active_synapses, run.progress, (run.log || "").length]);
  if (key === lastDetailKey) return;
  if (panel.contains(document.activeElement) && document.activeElement.tagName === "TEXTAREA") return;
  lastDetailKey = key;
  const rows = [];
  const add = (label, value) => value !== undefined && value !== null && value !== "" && rows.push(h("dt", { text: label }), h("dd", {}, [].concat(value)));
  add("Status", statusBadge(run));
  add("Kind", run.kind_label);
  add("Started", run.started ? new Date(run.started * 1000).toLocaleString() : null);
  add("Duration", duration(run.duration_s));
  add("Device", `${run.device === "auto" ? "GPU if available, else CPU" : "CPU"} · ${run.threads} threads`);
  add("Model", [run.family, run.task, run.dataset, run.strategy && `strategy ${run.strategy}`].filter(Boolean).join(" · "));
  if (run.summary) {
    add("Best epoch", `${run.summary.best_epoch} of ${run.summary.train_losses?.length ?? "?"} (lowest validation loss ${fmt(run.summary.best_loss, 4)})`);
  }
  if (run.resources) {
    const r = run.resources;
    add("Size", `${fmt(r.trainable_parameters)} trainable parameters · ${fmt(r.active_synapses)} active synapses${r.candidate_synapse_slots ? ` of ${fmt(r.candidate_synapse_slots)} candidate slots` : ""}`);
    if (r.persistent_state_scalars_per_sample) add("State", `${fmt(r.persistent_state_scalars_per_sample)} persistent state values per sample`);
  }
  add("Library", `dendritic_modeling ${run.core_version || "?"}`);
  add("Configuration", h("code", { text: `config.yaml · sha256 ${String(run.config_sha256 || "").slice(0, 12)}` }));
  add("Command", h("code", { class: "wrap", text: (run.command || []).join(" ") }));
  add("Folder", [h("code", { class: "wrap", text: run.cwd || "" }), " ", h("button", { type: "button", class: "linkish", text: "copy path", onclick: async () => toast((await copyText(run.cwd)) ? "Folder path copied." : "Copy failed; select the path instead.", { kind: "info" }) })]);
  if (run.error) add("Error", run.error);

  const notes = h("textarea", { class: "input", rows: 2, placeholder: "Notes about this run (saved with its record)" });
  notes.value = run.notes || "";
  notes.addEventListener("change", async () => {
    try {
      await api("annotate", { id: run.id, notes: notes.value });
      toast("Notes saved.", { kind: "ok" });
      emit("refresh-runs");
    } catch (error) {
      toast(error.message, { kind: "error" });
    }
  });

  const actions = h("div", { class: "actions" });
  if (run.status === "running") actions.append(h("button", { type: "button", class: "btn danger", onclick: () => stopRun(run) }, [icon("stop"), "Stop"]));
  actions.append(
    h("button", { type: "button", class: "btn", onclick: () => useConfiguration(run) }, [icon("upload"), "Open in editor"]),
    h("button", { type: "button", class: "btn", onclick: () => renameRun(run) }, [icon("gear"), "Rename"])
  );

  const synapses = run.resources?.synapses?.length ? pathwayTable(run.resources.synapses) : null;
  const log = h("details", { class: "log", open: run.status === "failed" || run.status === "failed to start" }, [
    h("summary", {}, [icon("chevron"), "Worker log", h("span", { class: "meta", text: "last 40 kB" })]),
    h("pre", { text: run.log || "(empty)" }),
  ]);
  panel.replaceChildren(
    h("div", { class: "panel-head" }, [h("h3", { text: "Run details" }), h("span", { class: "sub", text: runTitle(run) })]),
    h("div", { class: "panel-body" }, [h("dl", { class: "facts" }, rows), synapses, h("label", { class: "dialog-label", text: "Notes" }), notes, actions, log])
  );
  const pre = log.querySelector("pre");
  if (log.open) pre.scrollTop = pre.scrollHeight;
}

function pathwayTable(synapses) {
  const by = new Map();
  for (const sy of synapses) {
    const key = `${sy.pathway}|${sy.soma_relative_depth}`;
    const entry = by.get(key) || { pathway: sy.pathway, depth: sy.soma_relative_depth, active: 0, slots: 0 };
    entry.active += sy.active_synapses || 0;
    entry.slots += sy.candidate_slots || 0;
    by.set(key, entry);
  }
  const rows = [...by.values()].sort((a, b) => a.pathway.localeCompare(b.pathway) || a.depth - b.depth);
  return h("details", { class: "disclosure" }, [
    h("summary", {}, [icon("chevron"), "Synapses by pathway and level", h("span", { class: "meta", text: "measured on the built model" })]),
    h("table", { class: "data" }, [
      h("thead", {}, [h("tr", {}, [h("th", { text: "Pathway" }), h("th", { text: "Level (0 = soma)" }), h("th", { class: "num", text: "Active synapses" }), h("th", { class: "num", text: "Candidate slots" })])]),
      h("tbody", {}, rows.map(r => h("tr", {}, [h("td", { text: r.pathway.replace(/_/g, " ") }), h("td", { text: String(r.depth) }), h("td", { class: "num", text: fmt(r.active) }), h("td", { class: "num", text: fmt(r.slots) })]))),
    ]),
  ]);
}

async function stopRun(run) {
  try {
    await api("stop", { id: run.id });
    toast(`Stopped ${runTitle(run)}.`, { kind: "info" });
  } catch (error) {
    toast(error.message, { kind: "error" });
  }
  emit("refresh-runs");
}

async function renameRun(run) {
  const name = await promptDialog({ title: "Rename run", label: "Name", value: run.name || runTitle(run), confirm: "Save" });
  if (name === null) return;
  try {
    await api("annotate", { id: run.id, name });
    emit("refresh-runs");
  } catch (error) {
    toast(error.message, { kind: "error" });
  }
}

async function useConfiguration(run) {
  try {
    const config = await loadConfig(run.id);
    const result = await api("preview", { config });
    adopt(result, runTitle(run), { keepHistory: true });
    emit("show-view", "model");
    toast(`Opened the configuration of “${runTitle(run)}”. Undo returns to your previous design.`, { kind: "ok" });
  } catch (error) {
    toast(`This run's configuration no longer validates: ${error.message}`, { kind: "error" });
  }
}

function exportCurves() {
  const chosen = compareRuns(state.runs?.runs || []);
  const list = chosen.length ? chosen : (state.runs?.runs || []).filter(r => r.id === state.focusRun).map(r => ({ label: runTitle(r), train: r.summary?.train_losses || [], valid: r.summary?.valid_losses || [] }));
  download("dendrinet-curves.csv", curvesCsv(list), "text/csv");
}

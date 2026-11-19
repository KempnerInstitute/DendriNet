// DendriNet Studio: application shell, views, runs polling and actions.

import { h, s, icon } from "./dom.js";
import { state, api, on, adopt, undo, redo, readDraft, clearDraft, configKey } from "./store.js";
import { mountModel, renderModel, currentCheck } from "./design.js";
import { mountData, renderData, mountTraining, renderTraining, mountOptions, renderOptions } from "./forms.js";
import { mountYaml, refreshYaml } from "./yamlview.js";
import { mountRuns, updateRuns, runTitle } from "./runs.js";
import { toast, promptDialog, confirmDialog } from "./ui.js";

const VIEWS = [
  ["model", "Model", "network", mountModel, renderModel],
  ["data", "Data & task", "folder", mountData, renderData],
  ["training", "Training", "sliders", mountTraining, renderTraining],
  ["options", "All options", "gear", mountOptions, renderOptions],
  ["yaml", "YAML", "copy", mountYaml, () => refreshYaml()],
  ["runs", "Runs", "play", mountRuns, updateRuns],
];

const mounted = new Set();
const stale = new Set();
const ui = {};
let pollTimer = null;

function logo() {
  return s("svg", { class: "logo", width: 22, height: 22, viewBox: "0 0 24 24", "aria-hidden": "true" }, [
    s("path", { d: "M12 15.5V10M12 10L6 4.5M12 10L18 4.5M6 4.5L3.5 2M6 4.5L8 2M18 4.5L16 2M18 4.5L20.5 2", class: "logo-tree" }),
    s("path", { d: "M12 13.5L17 22H7Z", class: "logo-soma" }),
  ]);
}

function buildShell() {
  ui.docName = h("button", { type: "button", class: "doc-name", title: "Rename this design", onclick: renameDocument });
  ui.edited = h("span", { class: "edited", hidden: true }, [icon("dot"), "edited"]);
  ui.status = h("span", { class: "status", role: "status", "aria-live": "polite" });
  ui.tabs = h("nav", { class: "tabs", role: "tablist", "aria-label": "Views" });
  for (const [key, label] of VIEWS) {
    const tab = h("button", { type: "button", role: "tab", id: `tab-${key}`, "aria-controls": `view-${key}`, "aria-selected": "false", onclick: () => showView(key) }, [h("span", { text: label })]);
    if (key === "runs") tab.append((ui.runBadge = h("span", { class: "count", hidden: true })));
    ui.tabs.append(tab);
  }
  ui.undo = h("button", { type: "button", class: "icon-btn", "aria-label": "Undo (Ctrl+Z)", title: "Undo (Ctrl+Z)", onclick: () => undo() }, [icon("undo")]);
  ui.redo = h("button", { type: "button", class: "icon-btn", "aria-label": "Redo (Ctrl+Shift+Z)", title: "Redo (Ctrl+Shift+Z)", onclick: () => redo() }, [icon("redo")]);
  ui.device = h("select", { class: "input compact", "aria-label": "Device for runs" }, [h("option", { value: "cpu", text: "CPU" }), h("option", { value: "auto", text: "GPU if available" })]);
  ui.check = h("button", { type: "button", class: "btn", onclick: checkModel, title: "Build the model and count its parameters and synapses" }, [icon("check"), "Check model"]);
  ui.train = h("button", { type: "button", class: "btn primary", onclick: train, title: "Train this configuration in the background" }, [icon("play"), "Train"]);
  ui.designOnly = h("span", { class: "tag", hidden: true, title: "Started with --no-execution: export the YAML and run it with dendritic-train", text: "design only" });
  const header = h("header", { class: "topbar" }, [
    h("div", { class: "brand" }, [logo(), h("span", { class: "brand-name", text: "DendriNet" }), h("span", { class: "brand-app", text: "Studio" })]),
    h("div", { class: "doc" }, [ui.docName, ui.edited, ui.status, h("span", { class: "history" }, [ui.undo, ui.redo])]),
    ui.tabs,
    h("div", { class: "top-actions" }, [ui.designOnly, ui.device, ui.check, ui.train]),
  ]);
  ui.views = h("main", { class: "views" });
  for (const [key] of VIEWS) ui.views.append(h("section", { id: `view-${key}`, class: `view view-${key}`, role: "tabpanel", "aria-labelledby": `tab-${key}`, hidden: true }));
  ui.footer = h("footer", { class: "statusbar" });
  document.getElementById("app").replaceChildren(header, ui.views, ui.footer);
}

function renderHeader() {
  ui.docName.textContent = state.docName || "untitled.yaml";
  ui.edited.hidden = !state.dirty;
  ui.undo.disabled = !state.history.length;
  ui.redo.disabled = !state.future.length;
  let kind;
  let text;
  if (state.validating) [kind, text] = ["pending", "Checking…"];
  else if (state.error) [kind, text] = ["error", "Not valid"];
  else if (currentCheck()) [kind, text] = ["ok", "Valid · model checked"];
  else [kind, text] = ["ok", "Valid"];
  ui.status.className = `status ${kind}`;
  ui.status.title = state.error || "DendriNet's strict loader accepts this configuration.";
  ui.status.replaceChildren(icon(kind === "error" ? "cross" : kind === "pending" ? "dot" : "check"), text);
  const execution = state.schema?.execution !== false;
  const busy = Boolean(state.runs?.busy);
  ui.designOnly.hidden = execution;
  for (const el of [ui.device, ui.check, ui.train]) el.hidden = !execution;
  ui.check.disabled = busy || Boolean(state.error);
  ui.train.disabled = busy || Boolean(state.error);
  const why = state.error ? "Fix the configuration first" : busy ? "A run is in progress; one worker runs at a time" : null;
  ui.check.title = why || "Build the model and count its parameters and synapses";
  ui.train.title = why || "Train this configuration in the background";
  const running = (state.runs?.runs || []).filter(r => r.status === "running").length;
  ui.runBadge.hidden = !running;
  ui.runBadge.textContent = running ? `${running} running` : "";
}

function renderFooter() {
  const sc = state.schema;
  const runs = state.runs;
  ui.footer.replaceChildren(
    h("span", { text: `dendritic_modeling ${sc.core_version}` }),
    h("span", { text: `Studio ${sc.studio_version}` }),
    h("span", { text: sc.execution === false ? "design only" : `runs use ${runs?.threads ?? sc.threads} CPU threads` }),
    h("span", { class: "grow" }),
    h("span", { class: "path", title: "Workspace folder: every run is saved here", text: runs?.workspace || sc.workspace })
  );
}

export function showView(key) {
  if (!VIEWS.some(([k]) => k === key)) key = "model";
  state.view = key;
  for (const [k, , , mount, render] of VIEWS) {
    const tab = document.getElementById(`tab-${k}`);
    const panel = document.getElementById(`view-${k}`);
    const active = k === key;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    panel.hidden = !active;
    if (!active) continue;
    if (!mounted.has(k)) {
      mounted.add(k);
      stale.delete(k);
      mount(panel);
    } else if (stale.has(k)) {
      stale.delete(k);
      render();
    }
  }
  try {
    history.replaceState(null, "", `${location.pathname}${location.search}#${location.hash.slice(1).split("&")[0]}&view=${key}`);
  } catch (error) {
    /* the address bar is a convenience */
  }
}

function renderViews() {
  for (const [key, , , , render] of VIEWS) {
    if (!mounted.has(key)) continue;
    if (key === state.view) render();
    else stale.add(key);
  }
}

// Runs -----------------------------------------------------------------------------------

async function refreshRuns() {
  try {
    state.runs = await api("runs");
  } catch (error) {
    ui.status.title = `The local Studio service is not responding: ${error.message}`;
    schedulePoll(5000);
    return;
  }
  trackModelCheck();
  renderHeader();
  renderFooter();
  if (mounted.has("runs")) {
    if (state.view === "runs") updateRuns();
    else stale.add("runs");
  }
  const pending = state.modelCheck && !state.modelCheck.resources && !state.modelCheck.error;
  schedulePoll(state.runs.busy || pending ? 1500 : 8000);
}

function schedulePoll(ms) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(() => {
    if (document.hidden) schedulePoll(ms);
    else refreshRuns();
  }, ms);
}

function trackModelCheck() {
  const check = state.modelCheck;
  if (!check?.runId || check.resources || check.error) return;
  const run = state.runs.runs.find(r => r.id === check.runId);
  if (!run || run.status === "running") return;
  if (run.status === "completed" && run.resources) {
    check.resources = run.resources;
    if (check.key === configKey()) toast(`Model check: ${run.resources.trainable_parameters.toLocaleString("en-US")} trainable parameters, ${run.resources.active_synapses.toLocaleString("en-US")} active synapses.`, { kind: "ok" });
  } else {
    check.error = run.status;
    toast(`The model check did not complete (${run.status}).`, { kind: "error", action: { label: "Show log", run: () => {
      state.focusRun = run.id;
      showView("runs");
    } } });
  }
  renderViews();
}

function device() {
  return ui.device.value;
}

async function checkModel() {
  if (state.error) return toast(`Fix the configuration first: ${state.error}`, { kind: "error" });
  try {
    const run = await api("run", { config: state.config, kind: "inspect", device: device(), name: `Model check · ${state.docName}` });
    state.modelCheck = { key: configKey(), runId: run.id };
    toast("Building the model to count its parameters and synapses…", { kind: "info" });
    renderViews();
    refreshRuns();
  } catch (error) {
    toast(error.message, { kind: "error" });
  }
}

async function train() {
  if (state.error) return toast(`Fix the configuration first: ${state.error}`, { kind: "error" });
  const p = state.preview;
  const stem = (state.docName || "model").replace(/\.(ya?ml)$/i, "");
  const epochs = p.resolved_config.training?.main?.common?.epochs;
  const name = await promptDialog({
    title: "Train this model",
    label: "Run name",
    value: `${stem}${epochs ? ` · ${epochs} epochs` : ""}`,
    confirm: "Start training",
    help: `Runs on ${device() === "cpu" ? "the CPU" : "a GPU if one is available, else the CPU"} with ${state.runs?.threads ?? state.schema.threads} threads. The dataset ${p.dataset} is prepared (or downloaded) first. Everything is saved under ${state.runs?.workspace || state.schema.workspace}.`,
  });
  if (name === null) return;
  try {
    const run = await api("run", { config: state.config, kind: "train", device: device(), name });
    state.focusRun = run.id;
    showView("runs");
    toast(`Started “${runTitle(run)}”.`, { kind: "ok" });
    refreshRuns();
  } catch (error) {
    toast(error.message, { kind: "error" });
  }
}

async function renameDocument() {
  const name = await promptDialog({ title: "Rename this design", label: "File name", value: state.docName || "my_model.yaml", confirm: "Rename" });
  if (!name) return;
  state.docName = /\.ya?ml$/i.test(name) ? name : `${name}.yaml`;
  renderHeader();
}

// Start-up ----------------------------------------------------------------------------------

function keyboard(event) {
  const mod = event.ctrlKey || event.metaKey;
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "");
  if (mod && !typing && event.key.toLowerCase() === "z") {
    event.preventDefault();
    if (event.shiftKey) redo();
    else undo();
  } else if (mod && !typing && event.key.toLowerCase() === "y") {
    event.preventDefault();
    redo();
  } else if (event.altKey && /^[1-6]$/.test(event.key)) {
    event.preventDefault();
    showView(VIEWS[Number(event.key) - 1][0]);
  }
}

async function restoreDraft() {
  const draft = readDraft();
  if (!draft) return false;
  const when = new Date(draft.saved).toLocaleString();
  const restore = await confirmDialog({ title: "Unsaved design found", text: `This browser kept “${draft.name || "a design"}” from ${when}, which was never exported or trained. Restore it?`, confirm: "Restore", cancel: "Discard" });
  if (!restore) {
    clearDraft();
    return false;
  }
  try {
    const result = await api("preview", { config: draft.config });
    adopt(result, draft.name, { dirty: true });
    return true;
  } catch (error) {
    toast(`The saved design is not valid with this DendriNet version (${error.message}); it was not restored.`, { kind: "error", timeout: 12000 });
    return false;
  }
}

async function start() {
  buildShell();
  try {
    state.schema = await api("schema");
  } catch (error) {
    document.getElementById("app").replaceChildren(
      h("div", { class: "fatal" }, [
        h("h1", { text: "DendriNet Studio cannot reach its local service" }),
        h("p", { text: error.message }),
        h("p", { text: "Open the full address the dendrinet-studio command printed (it ends with a #token), and keep that command running." }),
      ])
    );
    return;
  }
  on("validating", renderHeader);
  on("validated", renderHeader);
  on("edited", renderHeader);
  on("saved", renderHeader);
  on("preview", () => {
    renderHeader();
    renderViews();
  });
  on("invalid", () => {
    renderHeader();
    renderViews();
  });
  on("selection", () => {
    if (mounted.has("model")) renderModel();
  });
  on("loaded", () => {
    renderHeader();
    for (const key of mounted) if (key !== state.view) stale.add(key);
    if (mounted.has("yaml")) refreshYaml(true);
  });
  on("show-view", showView);
  on("undo", () => undo());
  on("refresh-runs", refreshRuns);
  document.addEventListener("keydown", keyboard);
  document.addEventListener("visibilitychange", () => !document.hidden && refreshRuns());

  const restored = await restoreDraft();
  if (!restored) {
    const first = state.schema.examples[0];
    if (!first) {
      document.getElementById("app").append(h("p", { class: "fatal", text: "No example configurations are installed with this DendriNet version." }));
      return;
    }
    const result = await api(`example/${encodeURIComponent(first.name)}`);
    adopt(result, `${first.name}.yaml`);
  }
  document.getElementById("app").removeAttribute("aria-busy");
  renderHeader();
  renderFooter();
  const requested = /view=(\w+)/.exec(location.hash)?.[1];
  showView(requested || "model");
  refreshRuns();
}

start();

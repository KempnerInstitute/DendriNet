// Application state, the server API, edit history and validation.
//
// The configuration being edited is the single source of truth. Every edit
// goes through `edit()`, which records undo history and asks the library to
// validate the new configuration. The last valid inspection drives the
// drawings, so an invalid intermediate edit never blanks the screen.

// The launcher puts the session token in the address fragment (#token&view=...).
const token = location.hash.slice(1).split("&")[0];
const listeners = new Map();
// A host can serve several personal Studios, each under its own URL prefix.
const basePath = new URL(".", location.href).pathname;
const DRAFT_KEY = "dendrinet-studio-draft-v1" + (basePath === "/" ? "" : ":" + basePath);
const HISTORY_LIMIT = 100;

export const state = {
  schema: null,
  config: null,
  preview: null,
  error: null,
  errorPath: null,
  docName: "",
  dirty: false,
  validating: false,
  selection: { kind: "network" },
  view: "design",
  history: [],
  future: [],
  modelCheck: null,
  runs: null,
  selectedRuns: new Set(),
  focusRun: null,
  showKeys: false,
};

export function on(event, fn) {
  if (!listeners.has(event)) listeners.set(event, []);
  listeners.get(event).push(fn);
}

export function emit(event, detail) {
  for (const fn of listeners.get(event) || []) {
    try {
      fn(detail);
    } catch (error) {
      console.error(error);
    }
  }
}

export async function api(path, data, signal) {
  const response = await fetch("api/" + path, {
    method: data === undefined ? "GET" : "POST",
    headers: { "X-Studio-Token": token, "Content-Type": "application/json" },
    body: data === undefined ? undefined : JSON.stringify(data),
    signal,
  });
  let result;
  try {
    result = await response.json();
  } catch (error) {
    throw new Error(`The local Studio service replied without data (HTTP ${response.status})`);
  }
  if (!response.ok) throw new Error(result.error || response.statusText);
  return result;
}

// Paths -----------------------------------------------------------------

export function getPath(obj, path) {
  return path.split(".").reduce((o, k) => (o === undefined || o === null ? undefined : o[k]), obj);
}

export function setPath(obj, path, value) {
  const keys = path.split(".");
  let o = obj;
  for (const k of keys.slice(0, -1)) {
    if (o[k] === undefined || o[k] === null || typeof o[k] !== "object") o[k] = {};
    o = o[k];
  }
  o[keys.at(-1)] = value;
}

export function deletePath(obj, path) {
  const keys = path.split(".");
  const parent = getPath(obj, keys.slice(0, -1).join("."));
  if (parent && typeof parent === "object") delete parent[keys.at(-1)];
}

// Resolved value: what the configuration says, else the library's resolved value.
export function valueAt(path) {
  const own = getPath(state.config, path);
  if (own !== undefined) return own;
  return state.preview ? getPath(state.preview.resolved_config, path) : undefined;
}

// Population network helpers ------------------------------------------------

export const isPopulationNetwork = () => state.config?.model?.core?.type === "population_network";

export function network() {
  return state.config?.model?.core?.population_network;
}

export function layers() {
  return network()?.layers || [];
}

export function nodeView(layer, population) {
  return state.preview?.nodes?.find(n => n.layer === layer && n.population === population);
}

export function layerView(layer) {
  return state.preview?.layers?.[layer];
}

// Editing -------------------------------------------------------------------

export let validationRevision = 0;
let documentGeneration = 0;
let topologyQueue = Promise.resolve();
const pendingTopology = new Set();
let validatedDraft = Promise.resolve();
let finishValidation;

function validationFinished() {
  state.validating = false;
  finishValidation?.();
  finishValidation = undefined;
}

// Return synchronously when idle so Copy can use the browser's click gesture.
export function waitForTopologyEdits() {
  if (pendingTopology.size) return Promise.all(pendingTopology);
}

async function settleEdits() {
  do {
    await waitForTopologyEdits();
    while (state.validating) await validatedDraft;
  } while (pendingTopology.size);
}

export async function modelReady() {
  await settleEdits();
  if (state.error) throw new Error(state.error);
}

// Dialogs take their snapshot after earlier edits have reached the preview.
export async function topologyReady({ allowRepair = false } = {}) {
  const generation = documentGeneration;
  await settleEdits();
  if (generation !== documentGeneration) throw new Error("The design changed. Select the item and try again.");
  if (!allowRepair && state.error) throw new Error(state.error);
}

// Stage a topology edit on a copy. A rejected route must never become the
// document used by Train, by the YAML editor, or by draft recovery.
export function editTopology(mutate, { revision, signal, reindexes = false } = {}) {
  const generation = documentGeneration;
  const pending = topologyQueue.then(async () => {
    while (!signal?.aborted) {
      while (state.validating) await validatedDraft;
      if (signal?.aborted) return;
      if (generation !== documentGeneration || (revision !== undefined && revision !== validationRevision)) {
        throw new Error(revision === undefined ? "The design changed. Select the item and make the edit again." : "The design changed. Reopen this editor.");
      }
      // Bind confirmations to what the user saw. A matching revision may
      // contain the very error the proposed edit will repair.
      const base = validationRevision;
      const draft = structuredClone(state.config);
      const selection = mutate(draft);
      const before = JSON.stringify(state.config);
      if (JSON.stringify(draft) === before) {
        if (selection) select(selection);
        return;
      }
      let result;
      try {
        result = await api("preview", { config: draft }, signal);
      } catch (error) {
        if (signal?.aborted) return;
        if (base !== validationRevision) continue;
        throw error;
      }
      if (signal?.aborted) return;
      // A field can also change while this request is in flight. Rebuild ordinary
      // edits on the current draft. A dialog stays bound to the revision it showed.
      if (base !== validationRevision) continue;
      ++validationRevision;
      if (reindexes) ++documentGeneration;
      state.history.push(before);
      if (state.history.length > HISTORY_LIMIT) state.history.shift();
      state.future = [];
      state.config = draft;
      state.preview = result;
      state.error = state.errorPath = null;
      state.dirty = true;
      if (selection) state.selection = selection;
      fixSelection();
      saveDraft();
      emit("edited", { structural: true });
      emit("preview", { structural: true });
      emit("validated");
      return;
    }
  });
  // Waiting actions receive the rejection. A later edit can still be tried.
  pendingTopology.add(pending);
  const settled = () => { pendingTopology.delete(pending); };
  topologyQueue = pending.then(settled, settled);
  return pending;
}

export function edit(mutate, options = {}) {
  const before = JSON.stringify(state.config);
  mutate(state.config);
  const after = JSON.stringify(state.config);
  if (before === after) return Promise.resolve(false);
  state.history.push(before);
  if (state.history.length > HISTORY_LIMIT) state.history.shift();
  state.future = [];
  state.dirty = true;
  state.errorPath = options.path || null;
  saveDraft();
  emit("edited", options);
  return validate(options);
}

export function undo() {
  if (!state.history.length) return;
  ++documentGeneration;
  state.future.push(JSON.stringify(state.config));
  state.config = JSON.parse(state.history.pop());
  state.dirty = true;
  saveDraft();
  emit("edited", { structural: true });
  return validate({ structural: true });
}

export function redo() {
  if (!state.future.length) return;
  ++documentGeneration;
  state.history.push(JSON.stringify(state.config));
  state.config = JSON.parse(state.future.pop());
  state.dirty = true;
  saveDraft();
  emit("edited", { structural: true });
  return validate({ structural: true });
}

export async function validate(options = {}) {
  const revision = ++validationRevision;
  if (!state.validating) validatedDraft = new Promise(resolve => { finishValidation = resolve; });
  state.validating = true;
  emit("validating");
  try {
    const result = await api("preview", { config: state.config });
    if (revision !== validationRevision) return false;
    state.preview = result;
    state.error = null;
    state.errorPath = null;
    fixSelection();
    emit("preview", options);
    return true;
  } catch (error) {
    if (revision !== validationRevision) return false;
    state.error = error.message;
    emit("invalid", options);
    return false;
  } finally {
    if (revision === validationRevision) {
      validationFinished();
      emit("validated");
    }
  }
}

// Load a complete configuration (an example, an import or applied YAML).
export function adopt(result, name, { keepHistory = false, dirty = keepHistory } = {}) {
  if (keepHistory && state.config) {
    state.history.push(JSON.stringify(state.config));
    state.future = [];
  } else {
    state.history = [];
    state.future = [];
  }
  ++validationRevision;
  ++documentGeneration;
  validationFinished();
  state.config = structuredClone(result.config);
  state.preview = result;
  state.error = null;
  state.errorPath = null;
  state.docName = name || state.docName;
  state.dirty = dirty;
  state.modelCheck = null;
  state.selection = defaultSelection();
  if (dirty) saveDraft();
  else clearDraft();
  emit("loaded");
  emit("preview", { structural: true });
}

export function defaultSelection() {
  if (state.preview?.family === "population_network" && state.preview.nodes?.length) {
    const readout = state.preview.layers?.[0]?.readout;
    const node = state.preview.nodes.find(n => n.layer === 0 && n.name === readout) || state.preview.nodes[0];
    return { kind: "population", layer: node.layer, population: node.population };
  }
  return { kind: "model" };
}

function fixSelection() {
  const sel = state.selection;
  if (!state.preview) return;
  if (state.preview.family !== "population_network") {
    if (["population", "layer", "connection", "network"].includes(sel.kind)) state.selection = { kind: "model" };
    return;
  }
  if (sel.kind === "model") state.selection = defaultSelection();
  if (sel.kind === "population" && !nodeView(sel.layer, sel.population)) state.selection = defaultSelection();
  if (sel.kind === "layer" && !layerView(sel.layer)) state.selection = defaultSelection();
  if (sel.kind === "connection") {
    const list = layers()[sel.layer]?.connections || [];
    if (!list[sel.index]) state.selection = defaultSelection();
  }
}

export function select(selection) {
  state.selection = selection;
  emit("selection");
}

// Drafts: an unsaved design survives a reload of the page (this browser only).
function saveDraft() {
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify({ config: state.config, name: state.docName, saved: Date.now() }));
  } catch (error) {
    /* storage unavailable: drafts are a convenience */
  }
}

export function clearDraft() {
  try {
    localStorage.removeItem(DRAFT_KEY);
  } catch (error) {
    /* ignore */
  }
}

export function readDraft() {
  try {
    const draft = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
    return draft && draft.config ? draft : null;
  } catch (error) {
    return null;
  }
}

export function markSaved() {
  state.dirty = false;
  clearDraft();
  emit("saved");
}

// A short fingerprint of the configuration, to tell whether a model check
// still describes the current design.
export function configKey(config = state.config) {
  const text = JSON.stringify(config);
  let hash = 0;
  for (let i = 0; i < text.length; i++) hash = (Math.imul(31, hash) + text.charCodeAt(i)) | 0;
  return String(hash);
}

// The Data & task, Training and All options views. Every control is generated
// from the library's typed configuration; the curated pages only choose which
// options to show together and explain them.

import { h, icon, fmt } from "./dom.js";
import { state, api, edit, getPath, setPath, deletePath, valueAt, isPopulationNetwork, layers } from "./store.js";
import { field, controlFor, selectInput, segmented, sectionForm, disclosure, humanize, rebuildKeepingFocus } from "./fields.js";
import { promptDialog, toast } from "./ui.js";

const NET = "model.core.population_network";
const LOCAL_RULE_HINT = /local|dfa|feedback|hebb|three|four|five|stdp|hsic|perturb/i;

const sections = path => state.schema.sections[path];

function card(title, sub, children, { id } = {}) {
  return h("section", { class: "panel form-card", id }, [
    h("div", { class: "panel-head" }, [h("h3", { text: title }), sub ? h("span", { class: "sub", text: sub }) : null]),
    h("div", { class: "panel-body" }, [].concat(children).filter(Boolean)),
  ]);
}

function pick(path, names, options = {}) {
  const section = sections(path);
  if (!section) return null;
  return sectionForm(section, { base: path, only: names, columns: options.columns || 2 });
}

// Data & task ---------------------------------------------------------------------------

let dataRoot = null;

export function mountData(container) {
  dataRoot = container;
  renderData();
}

export function renderData() {
  if (!dataRoot || !state.preview) return;
  rebuildKeepingFocus(dataRoot, () => dataRoot.replaceChildren(h("div", { class: "form-page" }, [taskCard(), datasetCard(), splitCard(), evaluationCard()])));
}

function lossChoices(task) {
  const tasks = state.schema.loss_tasks || {};
  const fits = Object.keys(tasks).filter(name => !tasks[name].length || tasks[name].includes(task));
  const other = Object.keys(tasks).filter(name => !fits.includes(name));
  const groups = { [`Losses for ${task}`]: fits };
  if (other.length) groups["Other tasks"] = other;
  return groups;
}

function taskCard() {
  const task = valueAt("model.task") || "classification";
  const loss = valueAt("training.main.common.loss_function");
  const tasks = state.schema.loss_tasks || {};
  const mismatch = loss && tasks[loss]?.length && !tasks[loss].includes(task);
  const outputs = valueAt("model.decoder.params.output_dim");
  return card("Task", "what the model predicts", [
    field({
      label: "Task",
      control: segmented(task, state.schema.tasks.map(t => [t, t === "classification" ? "Classification" : "Regression"]), setTask, "Task"),
      key: "model.task",
      help: task === "classification" ? "Predict one class per sample; the decoder outputs one score per class." : "Predict real values; the decoder outputs one value per target.",
    }),
    h("div", { class: "form-grid cols-2" }, [
      field({
        label: task === "classification" ? "Number of classes" : "Number of outputs",
        control: controlFor({ name: "output_dim", kind: "int" }, outputs, v => edit(cfg => setPath(cfg, "model.decoder.params.output_dim", v), { path: "model.decoder.params.output_dim" })),
        key: "model.decoder.params.output_dim",
        help: "The decoder's output dimension; it must match the dataset.",
      }),
      field({
        label: "Loss function",
        control: selectInput(loss, [], v => edit(cfg => setPath(cfg, "training.main.common.loss_function", v), { path: "training.main.common.loss_function" }), { groups: lossChoices(task) }),
        key: "training.main.common.loss_function",
        error: mismatch ? `${loss} is a ${tasks[loss].join("/")} loss.` : state.errorPath === "training.main.common.loss_function" ? state.error : null,
      }),
    ]),
  ]);
}

function setTask(task) {
  const tasks = state.schema.loss_tasks || {};
  const loss = valueAt("training.main.common.loss_function");
  const switchLoss = task === "regression" && tasks[loss]?.length && !tasks[loss].includes(task);
  // The final measures follow the task: classification scores, or mean squared error.
  const regression = task === "regression";
  edit(cfg => {
    setPath(cfg, "model.task", task);
    if (switchLoss) setPath(cfg, "training.main.common.loss_function", "mse");
    for (const name of ["accuracy", "auc", "categorical_loglikelihood"]) setPath(cfg, `analysis.performance.params.${name}`, !regression);
    setPath(cfg, "analysis.performance.params.mse", regression);
  }, { path: "model.task", structural: true });
  const changes = [switchLoss ? `the loss from ${loss} to mse` : null, regression ? "the final measures to mse" : "the final measures to accuracy, AUC and log-likelihood"].filter(Boolean);
  toast(`Changed ${changes.join(" and ")}.`, { kind: "info" });
}

function datasetCard() {
  const name = valueAt("data.dataset_name");
  const base = `data.dataset_params.${name}`;
  const params = getPath(state.config, base);
  const groups = state.schema.dataset_groups || {};
  const where = Object.entries(groups).find(([, names]) => names.includes(name))?.[0] || "";
  const children = [
    field({
      label: "Dataset",
      control: selectInput(name, [], v => edit(cfg => setPath(cfg, "data.dataset_name", v), { path: "data.dataset_name", structural: true }), { groups }),
      key: "data.dataset_name",
      help: where ? `${where}.` : null,
    }),
  ];
  children.push(h("h4", { class: "eyebrow", text: `Options of ${name}` }));
  const grid = h("div", { class: "form-grid cols-2" });
  for (const [key, value] of Object.entries(params || {})) grid.append(inferredField(`${base}.${key}`, key, value));
  if (!grid.children.length) grid.append(h("p", { class: "muted", text: "None set: the dataset's defaults apply." }));
  children.push(grid);
  children.push(h("div", { class: "add-row" }, [h("button", { type: "button", class: "btn small", onclick: () => addDatasetOption(base) }, [icon("plus"), "Add an option"])]));
  children.push(h("p", { class: "help", text: "Dataset options are passed to the dataset's builder, which checks them when the data is built (at a model check or a training run)." }));
  children.push(disclosure("data-processing", "Processing", () => pick("data.processing", null) || h("span"), { meta: "flatten · normalize · label noise" }));
  return card("Data", "where the samples come from", children);
}

function inferredField(path, key, value) {
  const kind = typeof value === "boolean" ? "bool" : typeof value === "number" ? (Number.isInteger(value) ? "int" : "float") : typeof value === "string" ? "str" : Array.isArray(value) && value.every(v => typeof v === "number") ? "list" : "mapping";
  const spec = { name: key, kind, item: kind === "list" && value.some(v => !Number.isInteger(v)) ? "float" : "int" };
  return field({
    label: humanize(key),
    control: controlFor(spec, value, v => edit(cfg => setPath(cfg, path, v), { path })),
    key: path,
    inherited: false,
    onReset: () => edit(cfg => deletePath(cfg, path), { path, structural: true }),
    error: state.errorPath === path ? state.error : null,
    wide: kind === "mapping",
  });
}

async function addDatasetOption(base) {
  const name = await promptDialog({ title: "Add a dataset option", label: "Option name", placeholder: "n_samples", confirm: "Next" });
  if (!name) return;
  const text = await promptDialog({ title: `Value of ${name}`, label: "Value (YAML)", placeholder: "1024", confirm: "Add" });
  if (text === null) return;
  try {
    const { value } = await api("parse", { yaml: text });
    edit(cfg => setPath(cfg, `${base}.${name.trim()}`, value), { path: `${base}.${name.trim()}`, structural: true });
  } catch (error) {
    toast(error.message, { kind: "error" });
  }
}

function splitCard() {
  return card("Splits and seeds", "which samples are used for what", [
    pick("experiment", ["train_valid_split", "seed", "dataset_seed", "split_seed", "loader_seed"]),
    h("p", { class: "help", text: "Unset seeds derive from the experiment seed. The test split stays separate from training and validation." }),
  ]);
}

function evaluationCard() {
  const perf = sections("analysis.performance");
  const measures = sections("analysis.performance.params");
  const task = valueAt("model.task") || "classification";
  return card("Evaluation", "what is measured after training", [
    perf ? sectionForm(perf, { base: "analysis.performance", columns: 2 }) : null,
    measures ? h("h4", { class: "eyebrow", text: "Measures" }) : null,
    measures ? sectionForm(measures, { base: "analysis.performance.params", columns: 2 }) : null,
    h("p", { class: "help", text: task === "regression" ? "For regression, report mse (and cosine_similarity); the classification measures do not apply." : "Final measures are written to performance/final.json in the run folder and shown in Runs." }),
  ]);
}

// Training ------------------------------------------------------------------------------

let trainingRoot = null;

export function mountTraining(container) {
  trainingRoot = container;
  renderTraining();
}

export function renderTraining() {
  if (!trainingRoot || !state.preview) return;
  rebuildKeepingFocus(trainingRoot, () =>
    trainingRoot.replaceChildren(h("div", { class: "form-page" }, [strategyCard(), scheduleCard(), optimizerCard(), initializationCard(), regularizationCard(), runtimeCard()]))
  );
}

function strategyCard() {
  const strategy = valueAt("training.main.strategy");
  const own = sections(`training.main.strategies.${strategy}`);
  const local = LOCAL_RULE_HINT.test(strategy || "");
  const children = [pick("training.main", ["strategy"], { columns: 1 })];
  if (own && own.fields.length) children.push(disclosure(`strategy-${strategy}`, `Options of ${strategy}`, () => sectionForm(own, { base: `training.main.strategies.${strategy}`, columns: 2 }), { open: true }));
  const rule = sections("training.main.learning_strategy_config");
  if (rule) children.push(disclosure("local-rule", "Local learning rule", () => sectionForm(rule, { base: "training.main.learning_strategy_config", columns: 2 }), { open: local, meta: local ? "used by this strategy" : "for local-learning strategies" }));
  children.push(h("p", { class: "help", text: "standard trains with backpropagation. Other strategies (feedback alignment, local rules, staged or frozen training) are registered in DendriNet; options a strategy does not use are rejected when checked." }));
  return card("Learning strategy", "how the weights are updated", children);
}

function scheduleCard() {
  return card("Schedule", "how long and in what batches", [
    pick("training.main.common", ["epochs", "batch_size", "shuffle", "early_stopping", "patience", "load_best_state_dict", "lr_schedule", "lr_warmup_epochs", "grad_clip_value", "grad_clip_mode"]),
  ]);
}

function optimizerCard() {
  const groups = sections("training.main.common.param_groups");
  return card("Optimizer and learning rates", "rates per parameter group", [
    pick("training.main.optimizer", ["name", "weight_decay", "momentum", "betas", "eps"]),
    h("h4", { class: "eyebrow", text: "Learning rates" }),
    groups ? sectionForm(groups, { base: "training.main.common.param_groups", columns: 2 }) : null,
    h("p", { class: "help", text: "lr is the base rate. With split parameters, synapse weights (topk_lr), branch couplings (blocklinear_lr), dendritic gates (reactivation_lr) and the decoder (decoder_lr) use their own rates." }),
    disclosure("optimizer-all", "All optimizer options", () => sectionForm(sections("training.main.optimizer"), { base: "training.main.optimizer", exclude: ["name", "weight_decay", "momentum", "betas", "eps"], columns: 2 })),
  ]);
}

// Initialization differs by family; seeds and gate calibration are shared.
function initializationCard() {
  const children = [];
  const family = state.preview.family;
  if (isPopulationNetwork()) {
    children.push(h("p", { class: "help", text: "Set for every layer (as layer defaults). A population or layer can still override a value in the Model view." }));
    const grid = h("div", { class: "form-grid cols-2" });
    for (const name of ["topk_init_method", "weight_transform", "dbl_init_method", "reactivation_init_policy", "reactivation_init_m", "reactivation_init_b", "adaptive_initialization", "initial_child_conductance"]) grid.append(networkWideField(name));
    children.push(grid);
  } else if (state.preview.payload) {
    const payload = state.preview.payload;
    const spec = state.schema.payloads[payload] || sections(`model.core.${payload}`);
    const names = spec ? spec.fields.map(f => f.name).filter(n => /init|bias|seed|scale/.test(n)) : [];
    if (spec && names.length) {
      const defaults = Object.fromEntries(spec.fields.map(f => [f.name, f.default]));
      children.push(sectionForm(spec, { base: `model.core.${payload}`, only: names, columns: 2, readValue: (n, which) => (which === "own" ? getPath(state.config, `model.core.${payload}.${n}`) : defaults[n]) }));
    } else children.push(h("p", { class: "muted", text: `${family} has no initialization options beyond its seeds.` }));
  } else {
    children.push(pick("model.core.sparsity", ["init_method", "noise_level"]));
    children.push(pick("model.core.morphology", ["dbl_init_method", "weight_transform"]));
    children.push(pick("model.core.reactivation", ["init_policy", "init_m", "init_b"]));
    children.push(pick("model.core.implementation", ["adaptive_initialization", "adaptive_initialization_policy", "initial_child_conductance", "initial_conductance_scale"]));
  }
  children.push(h("h4", { class: "eyebrow", text: "Seeds" }));
  children.push(pick("experiment", ["model_seed", "topology_seed"]));
  children.push(pick("model.core", ["initialization_seed"]));
  if (state.preview.biological_neuron || isPopulationNetwork() || !state.preview.payload) {
    children.push(disclosure("gate-calibration", "Dendritic gate calibration", () => pick("training.main.common", ["reactivation_initialization_batch_size", "reactivation_initialization_num_batches", "reactivation_initialization_max_iterations", "reactivation_update_mode", "recalibrate_reactivation_every"]), { meta: "before the first epoch" }));
  }
  return card("Initialization", "how the network starts", children);
}

function popSpec(name) {
  return state.schema.payloads.population?.fields.find(f => f.name === name) || { name, kind: "any" };
}

function networkWideField(name) {
  const all = layers();
  const values = all.map(l => l.population_defaults?.[name]);
  const set = values.filter(v => v !== undefined);
  const same = set.length === all.length && set.every(v => JSON.stringify(v) === JSON.stringify(set[0]));
  const overrides = all.flatMap(l => l.populations.filter(p => p.population?.[name] !== undefined).map(p => `${l.name}.${p.name}`));
  const spec = popSpec(name);
  const shown = set.length ? set[0] : spec.default;
  const notes = [];
  if (set.length && !same) notes.push("Layers differ; a new value applies to every layer.");
  if (overrides.length) notes.push(`Overridden in ${overrides.join(", ")}.`);
  return field({
    label: humanize(name),
    control: controlFor(spec, shown, value =>
      edit(cfg => {
        for (const layer of getPath(cfg, NET).layers) {
          layer.population_defaults = layer.population_defaults || {};
          layer.population_defaults[name] = value;
        }
      }, { path: `${NET}.layers.0.population_defaults.${name}` })
    ),
    key: `${NET}.layers.*.population_defaults.${name}`,
    inherited: set.length ? (same ? false : "mixed") : "default",
    onReset: set.length
      ? () =>
          edit(cfg => {
            for (const layer of getPath(cfg, NET).layers) if (layer.population_defaults) delete layer.population_defaults[name];
          }, { structural: true })
      : null,
    help: notes.join(" ") || null,
  });
}

function regularizationCard() {
  return card("Regularization and pruning", null, [
    disclosure("regularization", "Regularization", () => sectionForm(sections("training.main.regularization"), { base: "training.main.regularization", columns: 2 })),
    disclosure("pruning", "Pruning", () => sectionForm(sections("training.main.pruning"), { base: "training.main.pruning", columns: 2 })),
    disclosure("synapse-dynamics", "Synapse decay and boosting", () => pick("training.main.common", ["synapse_decay_rate", "weight_boosting"])),
  ]);
}

function runtimeCard() {
  return card("Data loading and precision", "speed settings; they do not change the model", [
    pick("training.main.common", ["num_workers", "pin_memory", "persistent_workers", "use_amp", "enable_nan_checking"]),
    pick("experiment", ["deterministic", "allow_tf32", "float32_matmul_precision"]),
  ]);
}

// All options ------------------------------------------------------------------------------

let optionsRoot = null;
let optionsFilter = "";

export function mountOptions(container) {
  optionsRoot = container;
  renderOptions();
}

export function renderOptions() {
  if (!optionsRoot || !state.preview) return;
  rebuildKeepingFocus(optionsRoot, () => {
    const search = h("input", { type: "search", class: "input search", placeholder: "Search all options by name or YAML key", value: optionsFilter, "aria-label": "Search options", "data-key": "options-search" });
    const results = h("div", { class: "options-results" });
    const fill = () => {
      optionsFilter = search.value;
      results.replaceChildren(...optionList(optionsFilter));
    };
    search.addEventListener("input", fill);
    fill();
    optionsRoot.replaceChildren(
      h("div", { class: "form-page wide" }, [
        h("section", { class: "panel" }, [
          h("div", { class: "panel-head" }, [h("h3", { text: "All options" }), h("span", { class: "sub", text: "every setting of the configuration, read from the library's typed sections" })]),
          h("div", { class: "panel-body" }, [search, h("p", { class: "help", text: "A tag marks values that come from a default. Options of model families other than the selected one are listed too; the library rejects them for this family." })]),
        ]),
        results,
      ])
    );
  });
}

const TOP = ["experiment", "data", "model", "training", "analysis", "outputs", "distributed"];

function optionList(filter) {
  const term = filter.trim().toLowerCase();
  const all = state.schema.sections;
  if (term) {
    const matches = [];
    for (const [path, section] of Object.entries(all)) {
      if (!path) continue;
      const names = section.fields.filter(f => f.kind !== "section" && (`${path}.${f.name}`.toLowerCase().includes(term) || humanize(f.name).toLowerCase().includes(term))).map(f => f.name);
      if (names.length) matches.push(h("section", { class: "panel" }, [h("div", { class: "panel-head" }, [h("h3", { text: path })]), h("div", { class: "panel-body" }, [sectionForm(section, { base: path, only: names, columns: 2 })])]));
      if (matches.length >= 40) break;
    }
    return matches.length ? matches : [h("p", { class: "empty", text: "No option matches." })];
  }
  return TOP.map(top => {
    const paths = Object.keys(all).filter(p => p === top || p.startsWith(top + "."));
    return h("section", { class: "panel" }, [
      h("div", { class: "panel-head" }, [h("h3", { text: humanize(top) }), h("span", { class: "sub", text: `${paths.length} sections` })]),
      h("div", { class: "panel-body" }, paths.map(path => disclosure(`opt-${path}`, path, () => sectionForm(all[path], { base: path, columns: 2 }), { meta: setCount(path) }))),
    ]);
  });
}

function setCount(path) {
  const value = getPath(state.config, path);
  if (!value || typeof value !== "object") return "defaults";
  const n = Object.keys(value).length;
  return n ? `${fmt(n)} set` : "defaults";
}

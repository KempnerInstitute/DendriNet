// Form controls and the schema-driven section forms.
//
// A control commits on change (or Enter), never on every keystroke, and the
// panel holding it is not rebuilt by its own edit, so focus stays where it is.

import { h, icon, fmt } from "./dom.js";
import { state, api, edit, valueAt, getPath, setPath, deletePath } from "./store.js";
import { promptDialog } from "./ui.js";

let uid = 0;
const nextId = prefix => `${prefix}-${++uid}`;

// Field frame: label, control, YAML key, help and an inline error. ----------

// `inherited`: false when the configuration sets the value, true (or a short
// text such as "layer default") when a default applies.
export function field({ label, control, key, help, inherited, onReset, error, wide }) {
  const focusable = control.matches?.("input, select, textarea") ? control : control.querySelector?.("input, select, textarea") || control;
  const id = focusable.id || (focusable.id = nextId("f"));
  if (key) focusable.dataset.key = key;
  if (help) focusable.setAttribute("aria-describedby", id + "-help");
  const tag = inherited === true ? "default" : typeof inherited === "string" ? inherited : null;
  const head = h("div", { class: "field-head" }, [
    h("label", { for: id, text: label }),
    tag ? h("span", { class: "tag inherited", title: "Not set here: this value comes from a default", text: tag }) : null,
    inherited === false && onReset
      ? h("button", { class: "linkish", type: "button", title: "Remove this setting so the default applies", onclick: onReset, text: "reset" })
      : null,
  ]);
  const wrap = h("div", { class: "field" + (wide ? " wide" : "") + (error ? " has-error" : "") }, [
    head,
    control,
    key ? h("code", { class: "yaml-key", text: key }) : null,
    help ? h("p", { class: "help", id: id + "-help", text: help }) : null,
    error ? h("p", { class: "field-error", role: "alert" }, [icon("cross"), " ", error]) : null,
  ]);
  return wrap;
}

// Rebuild a panel without losing the control the user is in, or what they typed.
export function rebuildKeepingFocus(container, build) {
  const active = document.activeElement;
  const inside = active && container.contains(active);
  const key = inside ? active.dataset?.key : null;
  // Carry over only text the user typed and has not committed yet.
  const typed = inside && active.tagName !== "SELECT" && "value" in active && active.value !== active.defaultValue ? active.value : null;
  let range = null;
  try {
    if (inside && typeof active.selectionStart === "number") range = [active.selectionStart, active.selectionEnd];
  } catch (error) {
    range = null;
  }
  const scroll = container.scrollTop;
  build();
  container.scrollTop = scroll;
  if (!key) return;
  const next = [...container.querySelectorAll("[data-key]")].find(el => el.dataset.key === key);
  if (!next) return;
  if (typed !== null && next.value !== typed) next.value = typed;
  next.focus({ preventScroll: true });
  if (range && next.setSelectionRange) {
    try {
      next.setSelectionRange(...range);
    } catch (error) {
      /* number inputs have no selection */
    }
  }
}

function commitOn(input, read, onCommit) {
  const commit = () => {
    let value;
    try {
      value = read();
    } catch (error) {
      input.setCustomValidity(error.message);
      input.reportValidity();
      return;
    }
    input.setCustomValidity("");
    onCommit(value);
  };
  input.addEventListener("change", commit);
  input.addEventListener("keydown", event => {
    if (event.key === "Enter" && input.tagName !== "TEXTAREA") {
      event.preventDefault();
      commit();
    }
  });
}

export function textInput(value, onCommit, attrs = {}) {
  const input = h("input", { type: "text", class: "input", value: value ?? "", spellcheck: "false", ...attrs });
  commitOn(input, () => input.value, onCommit);
  return input;
}

export function numberInput(value, onCommit, { integer = false, min, max, step, optional = false } = {}) {
  const input = h("input", { type: "number", class: "input num", value: value ?? "", step: step ?? (integer ? 1 : "any"), min, max, inputmode: integer ? "numeric" : "decimal" });
  commitOn(
    input,
    () => {
      if (input.value.trim() === "") {
        if (optional) return null;
        throw new Error("Enter a number");
      }
      const v = Number(input.value);
      if (!Number.isFinite(v)) throw new Error("Enter a finite number");
      if (integer && !Number.isInteger(v)) throw new Error("Enter a whole number");
      if (min !== undefined && v < min) throw new Error(`Enter ${min} or more`);
      if (max !== undefined && v > max) throw new Error(`Enter ${max} or less`);
      return v;
    },
    onCommit
  );
  return input;
}

export function stepper(value, onCommit, { min = 0, max, label = "" } = {}) {
  const input = numberInput(value, onCommit, { integer: true, min, max });
  const bump = delta => () => {
    const next = Math.max(min, (Number(input.value) || 0) + delta);
    if (max !== undefined && next > max) return;
    input.value = next;
    onCommit(next);
  };
  const wrap = h("div", { class: "stepper" }, [
    h("button", { type: "button", "aria-label": `Decrease ${label}`, onclick: bump(-1) }, [icon("minus")]),
    input,
    h("button", { type: "button", "aria-label": `Increase ${label}`, onclick: bump(1) }, [icon("plus")]),
  ]);
  wrap.id = input.id = nextId("st");
  return wrap;
}

export function selectInput(value, choices, onCommit, { groups, allowCustom = true, labels = {} } = {}) {
  const select = h("select", { class: "input" });
  const add = (parent, v) => parent.append(h("option", { value: v, text: labels[v] || v }));
  const known = new Set();
  if (groups) {
    for (const [group, values] of Object.entries(groups)) {
      const og = h("optgroup", { label: group });
      for (const v of values) {
        add(og, v);
        known.add(v);
      }
      select.append(og);
    }
  } else {
    for (const c of choices) {
      add(select, c);
      known.add(c);
    }
  }
  const current = value === null || value === undefined ? "" : String(value);
  if (current && !known.has(current)) add(select, current);
  if (allowCustom) select.append(h("option", { value: "__custom__", text: "Other value…" }));
  select.value = current;
  select.addEventListener("change", async () => {
    if (select.value === "__custom__") {
      select.value = current;
      const custom = await promptDialog({ title: "Other value", label: "Value", value: current, help: "DendriNet checks the value when the configuration is validated." });
      if (custom === null || custom === "") return;
      onCommit(custom);
      return;
    }
    onCommit(select.value);
  });
  return select;
}

export function switchInput(value, onCommit, label) {
  const input = h("input", { type: "checkbox", role: "switch", class: "switch", "aria-label": label });
  input.checked = Boolean(value);
  input.addEventListener("change", () => onCommit(input.checked));
  return input;
}

export function segmented(value, options, onCommit, label) {
  const group = h("div", { class: "segmented", role: "radiogroup", "aria-label": label });
  for (const [v, text, cls] of options) {
    const button = h("button", {
      type: "button",
      role: "radio",
      class: cls || "",
      "aria-checked": String(v === value),
      text,
      onclick: () => {
        if (v !== value) onCommit(v);
      },
    });
    group.append(button);
  }
  group.id = nextId("seg");
  return group;
}

export function listInput(value, onCommit, { item = "int" } = {}) {
  const text = Array.isArray(value) ? value.join(", ") : "";
  const input = h("input", { type: "text", class: "input mono", value: text, placeholder: "comma-separated" });
  commitOn(
    input,
    () => {
      const parts = input.value.replace(/[[\]]/g, "").split(/[\s,]+/).filter(Boolean);
      return parts.map(p => {
        if (item === "str") return p;
        const v = Number(p);
        if (!Number.isFinite(v)) throw new Error(`"${p}" is not a number`);
        if (item === "int" && !Number.isInteger(v)) throw new Error(`"${p}" is not a whole number`);
        return v;
      });
    },
    onCommit
  );
  return input;
}

// Mappings are edited as YAML (or JSON) and parsed by the server's YAML reader.
export function mappingInput(value, onCommit) {
  const text = value === undefined || value === null ? "" : JSON.stringify(value, null, 2);
  const area = h("textarea", { class: "input mono", rows: Math.min(10, Math.max(3, text.split("\n").length)), spellcheck: "false" });
  area.defaultValue = text;
  area.addEventListener("change", async () => {
    try {
      const parsed = area.value.trim() ? (await api("parse", { yaml: area.value })).value : {};
      area.setCustomValidity("");
      onCommit(parsed);
    } catch (error) {
      area.setCustomValidity(error.message);
      area.reportValidity();
    }
  });
  return area;
}

// Morphology: one chip per level (soma → distal), with presets. -------------

export const MORPHOLOGY_PRESETS = [[], [2], [2, 2], [3, 3, 3], [3, 3, 3, 3], [10, 11], [120]];

export function factorEditor(factors, onCommit) {
  const wrap = h("div", { class: "factors" });
  factors.forEach((b, index) => {
    wrap.append(
      h("span", { class: "factor", title: `Level ${index + 1}: each compartment has ${b} children` }, [
        h("span", { class: "factor-value", text: b }),
        h("button", { type: "button", "aria-label": `Fewer branches at level ${index + 1}`, onclick: () => {
          const next = [...factors];
          if (next[index] > 1) next[index] -= 1;
          else next.splice(index, 1);
          onCommit(next);
        } }, [icon("minus")]),
        h("button", { type: "button", "aria-label": `More branches at level ${index + 1}`, onclick: () => {
          const next = [...factors];
          next[index] += 1;
          onCommit(next);
        } }, [icon("plus")]),
      ])
    );
  });
  wrap.append(h("button", { type: "button", class: "btn small", onclick: () => onCommit([...factors, 2]) }, [icon("plus"), "level"]));
  if (!factors.length) wrap.prepend(h("span", { class: "muted", text: "Soma only" }));
  const presets = h("div", { class: "presets", role: "group", "aria-label": "Morphology presets" });
  for (const preset of MORPHOLOGY_PRESETS) {
    presets.append(h("button", { type: "button", class: "chip mono", "aria-pressed": String(JSON.stringify(preset) === JSON.stringify(factors)), onclick: () => onCommit([...preset]), text: preset.length ? `[${preset.join(",")}]` : "soma only" }));
  }
  const box = h("div", { class: "factor-editor" }, [wrap, presets]);
  box.id = nextId("fe");
  return box;
}

// Schema-driven sections ------------------------------------------------------

const HUMAN = {
  lr: "learning rate",
  lr_schedule: "learning-rate schedule",
  num_workers: "data-loader workers",
  grad_clip_value: "gradient clipping",
  dbl_init_method: "branch-coupling initialization",
  topk_init_method: "synapse initialization",
  reactivation_init_policy: "gate initialization",
  use_shunting: "shunting integration",
};

export function humanize(name) {
  if (HUMAN[name]) return capitalize(HUMAN[name]);
  return capitalize(name.replace(/_/g, " ").replace(/\blr\b/g, "learning rate").replace(/\bff\b/g, "feedforward").replace(/\brec\b/g, "recurrent"));
}

const capitalize = text => text.charAt(0).toUpperCase() + text.slice(1);

export function controlFor(spec, value, onCommit) {
  const label = humanize(spec.name);
  switch (spec.kind) {
    case "bool":
      return switchInput(value ?? spec.default, onCommit, label);
    case "int":
      return numberInput(value, onCommit, { integer: true, optional: spec.optional });
    case "float":
      return numberInput(value, onCommit, { optional: spec.optional });
    case "choice":
      return selectInput(value ?? spec.default, spec.choices || [], onCommit);
    case "list":
      return listInput(value ?? spec.default, onCommit, { item: spec.item === "str" ? "str" : spec.item === "float" ? "float" : "int" });
    case "mapping":
    case "any":
      return mappingInput(value ?? spec.default, onCommit);
    default:
      return textInput(value ?? "", v => onCommit(spec.optional && v === "" ? null : v));
  }
}

// One configuration section as a form. `base` is the YAML path of the section;
// `prefix` maps a field to where it is written (defaults to base + name).
export function sectionForm(section, { base, exclude = [], only, filter = "", columns = 1, readValue, write, remove } = {}) {
  const grid = h("div", { class: "form-grid cols-" + columns });
  const term = filter.trim().toLowerCase();
  for (const spec of section.fields) {
    if (exclude.includes(spec.name) || (only && !only.includes(spec.name))) continue;
    if (spec.kind === "section") continue;
    const path = base ? `${base}.${spec.name}` : spec.name;
    if (term && !(spec.name.includes(term) || humanize(spec.name).toLowerCase().includes(term))) continue;
    const own = readValue ? readValue(spec.name, "own") : getPath(state.config, path);
    const shown = own !== undefined ? own : readValue ? readValue(spec.name, "resolved") : valueAt(path);
    const commit = v => (write ? write(spec.name, v) : edit(cfg => setPath(cfg, path, v), { path }));
    const reset = () => (remove ? remove(spec.name) : edit(cfg => deletePath(cfg, path), { path, structural: true }));
    grid.append(
      field({
        label: humanize(spec.name),
        control: controlFor(spec, shown, commit),
        key: path,
        inherited: own === undefined,
        onReset: own !== undefined ? reset : null,
        error: state.errorPath === path ? state.error : null,
        help: spec.kind === "choice" && spec.choices?.length ? null : spec.optional && shown === null ? "Not set: the library decides." : null,
        wide: spec.kind === "mapping" || spec.kind === "any",
      })
    );
  }
  if (!grid.children.length) grid.append(h("p", { class: "muted", text: term ? "No matching option." : "This section has no options." }));
  return grid;
}

// Wraps a block in a disclosure that remembers whether it was open.
const openState = new Map();
export function disclosure(id, title, build, { open = false, meta } = {}) {
  const details = h("details", { class: "disclosure", "data-id": id });
  details.open = openState.has(id) ? openState.get(id) : open;
  const summary = h("summary", {}, [icon("chevron"), h("span", { text: title }), meta ? h("span", { class: "meta", text: meta }) : null]);
  details.append(summary);
  let built = false;
  const ensure = () => {
    if (!built && details.open) {
      details.append(build());
      built = true;
    }
  };
  details.addEventListener("toggle", () => {
    openState.set(id, details.open);
    ensure();
  });
  ensure();
  return details;
}

export function describeCount(n, one, many) {
  return `${fmt(n)} ${n === 1 ? one : many || one + "s"}`;
}

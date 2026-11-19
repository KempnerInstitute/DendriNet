// The YAML view: the exact file the Studio runs, editable as text. Applying
// validates the text with the library first; an invalid text never replaces
// the design, and the line the reader stopped at is marked.

import { h, icon, download, copyText } from "./dom.js";
import { state, api, adopt } from "./store.js";
import { toast } from "./ui.js";

let root = null;
let area = null;
let gutter = null;
let status = null;
let baseline = "";

export function mountYaml(container) {
  root = container;
  area = h("textarea", { class: "yaml-text", spellcheck: "false", wrap: "off", "aria-label": "Configuration YAML" });
  gutter = h("div", { class: "yaml-gutter", "aria-hidden": "true" });
  status = h("div", { class: "yaml-status", role: "status" });
  const file = h("input", { type: "file", accept: ".yaml,.yml,.json,text/yaml", hidden: true });
  file.addEventListener("change", async () => {
    const chosen = file.files?.[0];
    if (!chosen) return;
    area.value = await chosen.text();
    file.value = "";
    syncGutter();
    await apply(chosen.name.replace(/\.(ya?ml|json)$/i, ""));
  });
  root.replaceChildren(
    h("div", { class: "yaml-layout" }, [
      h("div", { class: "panel yaml-panel" }, [
        h("div", { class: "panel-head" }, [
          h("h3", { text: "Configuration file" }),
          h("span", { class: "sub", text: "This is exactly what Train runs and what dendritic-train reads." }),
          h("span", { class: "spacer" }),
          h("button", { type: "button", class: "btn", onclick: () => file.click() }, [icon("upload"), "Import"]),
          h("button", { type: "button", class: "btn", onclick: exportYaml }, [icon("download"), "Export"]),
          h("button", { type: "button", class: "btn", onclick: async () => toast((await copyText(area.value)) ? "YAML copied." : "Copy failed.", { kind: "info" }) }, [icon("copy"), "Copy"]),
        ]),
        h("div", { class: "yaml-editor" }, [gutter, area]),
        h("div", { class: "panel-foot" }, [
          status,
          h("span", { class: "spacer" }),
          h("button", { type: "button", class: "btn", id: "yaml-revert", onclick: () => refreshYaml(true) }, ["Revert"]),
          h("button", { type: "button", class: "btn primary", id: "yaml-apply", onclick: () => apply() }, [icon("check"), "Apply"]),
        ]),
        file,
      ]),
      h("aside", { class: "panel yaml-help" }, [
        h("div", { class: "panel-head" }, [h("h3", { text: "Working with the file" })]),
        h("div", { class: "panel-body" }, [
          h("p", { text: "Edit any option here, including options the forms do not show. Apply checks the text with DendriNet's strict loader: unknown keys and options the chosen model would ignore are reported, not dropped." }),
          h("p", { text: "Import accepts one standalone experiment YAML. Sweep files and files that extend another file are composed by dendritic-train; open the composed file instead." }),
          h("p", {}, ["Run an exported file without the Studio: ", h("code", { text: "dendritic-train my_model.yaml" })]),
          h("p", { class: "help", text: "Keyboard: Ctrl+Enter applies the text. Tab inserts two spaces." }),
        ]),
      ]),
    ])
  );
  area.addEventListener("input", () => {
    syncGutter();
    markChanged();
  });
  area.addEventListener("scroll", () => (gutter.scrollTop = area.scrollTop));
  area.addEventListener("keydown", event => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      apply();
    } else if (event.key === "Tab" && !event.shiftKey) {
      event.preventDefault();
      const { selectionStart: a, selectionEnd: b } = area;
      area.setRangeText("  ", a, b, "end");
      syncGutter();
      markChanged();
    }
  });
  refreshYaml(true);
}

// Show the current configuration (unless the text holds unapplied edits).
export async function refreshYaml(force = false) {
  if (!area) return;
  if (!force && area.value !== baseline) return;
  let text = state.preview && !state.error ? state.preview.yaml : null;
  if (text === null || text === undefined || JSON.stringify(state.config) !== JSON.stringify(state.preview?.config)) {
    try {
      text = (await api("dump", { config: state.config })).yaml;
    } catch (error) {
      text = JSON.stringify(state.config, null, 2);
    }
  }
  baseline = text;
  area.value = text;
  syncGutter();
  markChanged();
}

function syncGutter(errorLine = null) {
  const lines = area.value.split("\n").length;
  const frag = document.createDocumentFragment();
  for (let i = 1; i <= lines; i++) frag.append(h("div", { class: i === errorLine ? "error-line" : "", text: String(i) }));
  gutter.replaceChildren(frag);
  gutter.scrollTop = area.scrollTop;
}

function markChanged() {
  const changed = area.value !== baseline;
  root.querySelector("#yaml-apply").disabled = !changed;
  root.querySelector("#yaml-revert").disabled = !changed;
  if (changed) setStatus("warn", "Edited: apply to check and use this text.");
  else if (state.error) setStatus("error", state.error);
  else setStatus("ok", "Matches the current design.");
}

function setStatus(kind, text) {
  status.className = `yaml-status ${kind}`;
  status.replaceChildren(icon(kind === "ok" ? "check" : kind === "error" ? "cross" : "warn"), h("span", { text }));
}

async function apply(name) {
  setStatus("warn", "Checking…");
  try {
    const result = await api("preview", { yaml: area.value });
    adopt(result, name || state.docName, { keepHistory: true });
    baseline = area.value;
    syncGutter();
    markChanged();
    toast("Configuration applied. Undo restores the previous design.", { kind: "ok" });
  } catch (error) {
    const match = /line (\d+)/.exec(error.message);
    syncGutter(match ? Number(match[1]) : null);
    setStatus("error", error.message);
    if (match) {
      const line = Number(match[1]);
      const offset = area.value.split("\n").slice(0, line - 1).join("\n").length + (line > 1 ? 1 : 0);
      area.focus();
      area.setSelectionRange(offset, offset);
    }
  }
}

function exportYaml() {
  const name = (state.docName || "dendrinet_model").replace(/\.(ya?ml)$/i, "").replace(/[^\w.-]+/g, "_");
  download(`${name}.yaml`, area.value, "text/yaml");
  if (area.value !== baseline) toast("Exported the edited text, which has not been applied or checked.", { kind: "warn" });
}

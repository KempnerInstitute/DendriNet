import { h } from "./dom.js";
import { field, selectInput } from "./fields.js";
import { levelName } from "./morphology.js";

// The dialog presents depths from the soma outward. Only the YAML list is reversed.
export function travelEditor(original) {
  const number = value => h("input", { class: "input num", type: "number", min: 1, step: 1, value, required: true });
  const uniform = number(original?.delay_steps || 1);
  const individual = [];
  const mode = selectInput(original?.delay_steps_by_level ? "levels" : "uniform", ["uniform", "levels"], render,
    { allowCustom: false, labels: { uniform: "Same delay at every depth", levels: "Set delay by depth" } });
  const uniformField = field({ label: "Travel delay (steps)", control: uniform });
  const levels = h("div", { class: "form-grid cols-2" });
  const element = h("fieldset", { class: "travel-controls" }, [
    field({ label: "Travel delay", control: mode }),
    uniformField, levels,
    h("p", { class: "help", text: "A signal emitted on update 0 arrives on update N at a contacted depth with N steps of travel." }),
  ]);
  let target;
  function render() {
    const custom = mode.value === "levels";
    uniform.disabled = uniformField.hidden = custom;
    levels.hidden = !custom;
    levels.replaceChildren();
    if (!target || !custom) return;
    const deepest = target.branch_factors.length;
    for (let d = 0; d <= deepest; d++) {
      individual[d] ||= number(original?.delay_steps_by_level?.at(-1 - d) ?? (uniform.valueAsNumber || 1));
      levels.append(field({ label: "Travel steps at " + levelName(d, deepest), control: individual[d] }));
    }
  }
  function steps(input) {
    const n = input.valueAsNumber;
    if (!Number.isSafeInteger(n) || n < 1) throw new Error("Travel delays must be positive whole numbers of updates.");
    return n;
  }
  return {
    element,
    show(receiver, delayed) {
      target = receiver;
      element.disabled = element.hidden = !delayed || !target;
      render();
    },
    save(route) {
      if (element.disabled) {
        if (route.delay_steps !== 0) delete route.delay_steps;
        delete route.delay_steps_by_level;
      } else if (mode.value === "levels") {
        route.delay_steps_by_level = individual.slice(0, target.branch_factors.length + 1).map(steps).reverse();
        delete route.delay_steps;
      } else {
        const value = steps(uniform);
        if (value !== 1 || original?.delay_steps != null) route.delay_steps = value;
        else delete route.delay_steps;
        delete route.delay_steps_by_level;
      }
    },
  };
}

export function travelRange(edge, target) {
  const delays = edge.arrival_steps.filter((_, depth) =>
    (edge.target_depths == null || edge.target_depths.includes(depth)) && target.received_per_level[edge.pathway][depth]);
  if (!delays.length || edge.enabled === false) return null;
  const min = Math.min(...delays), max = Math.max(...delays);
  return min === max ? String(min) : `${min}–${max}`;
}

export function travelLabel(range) {
  return range === "1" ? "Δt" : range + " Δt";
}

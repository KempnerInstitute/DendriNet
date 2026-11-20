// A route belongs to its receiving layer. Edit the endpoints and the timing
// together so the core validates one complete proposal.
import { h } from "./dom.js";
import { field } from "./fields.js";
import { afferentMode } from "./afferents.js";
import { state, api, editTopology, validationRevision } from "./store.js";
import { dialog } from "./ui.js";
import { travelEditor } from "./travel.js";
import { targetDepths } from "./morphology.js";

const option = (value, label) => h("option", { value, text: label });
const selectBox = () => h("select", { class: "input" });
const externalNames = ["input", "input_e", "input_i"];

export async function connectionEditor(li, { index, from, to } = {}) {
  const revision = validationRevision;
  // A source can repair uncovered contacts. Derive its endpoints and budgets
  // from this draft, then submit the proposed connection to full validation.
  const view = state.error ? await api("wiring", { config: state.config }) : state.preview;
  if (revision !== validationRevision || !view.layers[li]) throw new Error("The design changed. Reopen the connection editor.");
  const original = view.edges.filter(e => e.layer === li)[index];
  if (index !== undefined && !original) throw new Error("The connection changed. Select it again.");
  const initialTarget = original?.target_id || to || (from ? "" : (state.selection.kind === "population" && state.selection.layer === li
    ? view.nodes.find(n => n.layer === li && n.population === state.selection.population)?.id
    : view.layers[li].name + "." + view.layers[li].readout));
  const targets = selectBox();
  targets.append(option("", "Choose a receiving population"));
  for (const layer of view.layers) {
    const group = h("optgroup", { label: layer.name });
    for (const n of view.nodes.filter(n => n.layer_name === layer.name)) group.append(option(n.id, n.id + " · " + n.polarity));
    targets.append(group);
  }
  targets.value = initialTarget || "";
  const timing = selectBox();
  timing.append(option("same_step", "Same step · feedforward"), option("delayed", "Delayed · recurrent"));
  const start = view.nodes.find(n => n.id === from);
  const end = view.nodes.find(n => n.id === to);
  const needsDelay = start && end && (start.layer > end.layer || start.id === end.id);
  timing.value = original?.timing || (needsDelay ? "delayed" : "same_step");
  const sources = selectBox();
  const travel = travelEditor(original);
  const sign = selectBox();
  const afferents = selectBox();
  afferents.append(option("", "Choose input organization"), option("paired", "Distinct E and I afferents with matched features"), option("split", "Separate groups of input channels"));
  const afferentField = field({ label: "External afferents for both signs", control: afferents,
    help: "This biological network needs separate sources for excitatory and inhibitory input. This setting applies to the first layer." });
  const summary = h("p", { class: "route-summary", role: "status" });
  const contacts = h("input", { type: "number", class: "input", min: 1, step: 1, value: 1 });
  const budgetField = field({ label: "Contacts per eligible compartment", control: contacts,
    help: "The target currently has zero contacts on this pathway. This budget is shared by its sources." });
  let selection = original ? (externalNames.includes(original.source) ? original.source : original.source_id) : from || "";
  let explicitSign = original && state.config.model.core.population_network.layers[li].connections[index].pathway
    ? original.pathway.split("_")[1] : "";
  let target, pathway, delayed, eligible;
  function refreshSources() {
    target = view.nodes.find(n => n.id === targets.value);
    delayed = timing.value === "delayed";
    travel.show(target, delayed);
    sources.replaceChildren(option("", "Choose a source"));
    if (target && !delayed) {
      const external = h("optgroup", { label: target.layer ? "Layer input" : "External input" });
      if (target.layer) {
        const prev = view.layers[target.layer - 1];
        external.append(option("input", "Previous readout · " + prev.name + "." + prev.readout));
      } else {
        external.append(option("input", "Input features"), option("input_e", "Excitatory afferents"), option("input_i", "Inhibitory afferents"));
      }
      sources.append(external);
    }
    for (const [i, layer] of view.layers.entries()) {
      if (target && !delayed && i > target.layer) continue;
      const candidates = view.nodes.filter(n => n.layer === i && (delayed || n.id !== target?.id));
      if (candidates.length) sources.append(h("optgroup", { label: layer.name },
        candidates.map(n => option(n.id, n.id + " · " + n.polarity))));
    }
    sources.value = [...sources.options].some(o => o.value === selection) ? selection : "";
    selection = sources.value;
    updatePathway();
  }
  function updatePathway() {
    const n = view.nodes.find(n => n.id === sources.value);
    const previous = target?.layer > 0 ? view.layers[target.layer - 1] : null;
    const inputPolarity = previous && sources.value === "input"
      ? view.nodes.find(p => p.layer_name === previous.name && p.name === previous.readout)?.polarity : null;
    const natural = n?.polarity || inputPolarity || (sources.value === "input_i" ? "inhibitory" : "excitatory");
    sign.replaceChildren(option("", sources.value ? "Automatic · " + natural : "Automatic · source polarity"));
    for (const polarity of ["excitatory", "inhibitory"]) {
      if (!view.biological_neuron || polarity === natural || explicitSign === polarity)
        sign.append(option(polarity, "Explicit · " + polarity));
    }
    sign.value = explicitSign;
    pathway = (delayed ? "rec_" : "ff_") + (sign.value || natural);
    const addingExternal = target?.layer === 0 && externalNames.includes(sources.value) && original?.enabled !== false;
    const otherSign = natural === "inhibitory" ? "ff_excitatory" : "ff_inhibitory";
    const needsAfferents = addingExternal && view.biological_neuron &&
      afferentMode(state.config.model.core.population_network) === "shared" &&
      view.edges.some(e => e !== original && e.layer === 0 && e.enabled !== false && externalNames.includes(e.source_id) && e.pathway === otherSign);
    afferentField.hidden = !needsAfferents;
    afferents.disabled = !needsAfferents;
    const available = target?.configured_per_level[pathway] || [];
    // A disabled soma and levels that an edited projection masks out cannot
    // contribute contacts, even if the underlying population budget is nonzero.
    const depths = targetDepths(original?.target_levels, available.length);
    eligible = available.map((_, d) => (d > 0 || target.settings.somatic_synapses) && (depths == null || depths.includes(d)));
    const active = original?.enabled !== false && original?.probability !== 0;
    const needsContacts = active && target && sources.value && eligible.some(Boolean) && !available.some((v, d) => eligible[d] && v > 0);
    budgetField.hidden = !needsContacts;
    contacts.disabled = !needsContacts;
    const notes = [];
    if (target && sources.value) notes.push(sources.value + " → " + target.id + ".");
    if (delayed) {
      notes.push("Uses earlier source activity. The source is zero before the sequence starts.");
      if (target && !view.layers[target.layer].recurrent) notes.push("This will enable recurrence on " + target.layer_name + ".");
    } else notes.push("Uses the current input or population response. Same-step routes must be acyclic.");
    if (target) {
      const otherSources = new Set(view.edges.filter(e => e !== original && e.enabled !== false &&
        e.target_id === target.id && e.pathway === pathway).map(e => e.source_id));
      if (otherSources.size) notes.push("Existing " + pathway.replace("_", " ") + " sources: " +
        [...otherSources].join(", ") + ". Sources on this pathway share the target's contact budget.");
    }
    if (target && sources.value && !eligible.some(Boolean)) notes.push("No receiving compartment is enabled. Check somatic contacts and target levels.");
    summary.textContent = notes.join(" ");
  }
  targets.onchange = timing.onchange = refreshSources;
  sources.onchange = () => { selection = sources.value; explicitSign = ""; updatePathway(); };
  sign.onchange = () => { explicitSign = sign.value; updatePathway(); };
  refreshSources();
  return dialog({
    title: original ? "Edit connection" : "Connect populations",
    body: [field({ label: "Receiving population", control: targets }),
      field({ label: "Timing", control: timing }), field({ label: "Source", control: sources }), travel.element,
      field({ label: "Pathway sign", control: sign }), afferentField, budgetField, summary],
    actions: [{ label: "Cancel", submit: false }, { label: original ? "Save connection" : "Add connection", primary: true, value: true }],
    onConfirm: async signal => {
      if (!target || !sources.value) throw new Error("Choose a receiving population and a source.");
      if (!afferents.disabled && !afferents.value) throw new Error("Choose how the two external afferent populations receive their features.");
      const n = view.nodes.find(n => n.id === sources.value);
      const source = n?.layer === target.layer ? n.name : sources.value;
      if (!eligible.some(Boolean) && original?.enabled !== false) throw new Error("This destination has no eligible compartment. Enable somatic contacts or change the target levels first.");
      if (!contacts.disabled && (!Number.isSafeInteger(contacts.valueAsNumber) || contacts.valueAsNumber < 1))
        throw new Error("Enter a positive whole number of contacts.");
      await editTopology(cfg => {
        const net = cfg.model.core.population_network;
        if (!afferents.disabled) {
          net.use_transfer = true;
          const transfer = net.transfer_params ||= {};
          transfer.independent_pathways = afferents.value === "split";
          transfer.input_source_identity = afferents.value === "paired" ? "paired_afferents" : "shared";
        }
        const dst = net.layers[target.layer];
        const list = dst.connections ||= [];
        const route = original ? net.layers[li].connections.splice(index, 1)[0] : {};
        if (!original || original.source !== source) route.source = source;
        route.target = target.name;
        if (delayed || "timing" in route) route.timing = timing.value;
        travel.save(route);
        if (sign.value) route.pathway = pathway;
        else delete route.pathway;
        if (delayed) dst.recurrent = true;
        if (!contacts.disabled) {
          const own = dst.populations[target.population].population ||= {};
          const key = pathway + "_synapses";
          const levels = target.configured_per_level[pathway];
          if (target.settings[key + "_by_level"]?.length || eligible.includes(false)) {
            own[key + "_by_level"] = levels.map((v, d) => eligible[d] ? contacts.valueAsNumber : v).reverse();
          } else own[key] = contacts.valueAsNumber;
        }
        const pos = original && li === target.layer ? index : list.length;
        list.splice(pos, 0, route);
        return { kind: "connection", layer: target.layer, index: pos };
      }, { revision, signal, reindexes: original && li !== target.layer });
    },
  });
}

// Population input pathways. The preview resolves names, signs, timing and level masks
// with the library; edits operate on the original YAML so imported masks survive.
import { h, fmt } from "./dom.js";
import { state, getPath, edit, select } from "./store.js";
import { field, selectInput, switchInput } from "./fields.js";
import { confirmDialog } from "./ui.js";

const NET = "model.core.population_network";
const external = new Set(["input", "input_e", "input_i"]);
const modes = {
  shared: "Shared input channels",
  paired: "Distinct E and I afferents with matched features",
  split: "Separate groups of input channels",
};

export function afferentMode(net) {
  if (!net.use_transfer) return "shared";
  if (net.transfer_params?.independent_pathways) return "split";
  return net.transfer_params?.input_source_identity === "paired_afferents" ? "paired" : "shared";
}

function inputIdentity() {
  const net = getPath(state.config, NET);
  const mode = afferentMode(net);
  const control = selectInput(mode, Object.keys(modes), async value => {
    const params = net.transfer_params || {};
    const splitKeys = value === "split" ? [] : ["excitatory_dim", "inhibitory_dim", "split_strategy", "split_seed"].filter(k => k in params);
    if (splitKeys.length && !(await confirmDialog({
      title: "Change the external afferents",
      text: "Leaving the channel split removes its settings: " + splitKeys.join(", ") + ". Undo restores them.",
      confirm: "Change afferents",
    }))) {
      control.value = mode;
      return;
    }
    edit(cfg => {
      const network = getPath(cfg, NET);
      network.use_transfer = true;
      const transfer = network.transfer_params ||= {};
      for (const key of splitKeys) delete transfer[key];
      transfer.independent_pathways = value === "split";
      transfer.input_source_identity = value === "paired" ? "paired_afferents" : "shared";
    }, { path: NET + ".transfer_params", structural: true });
  }, { allowCustom: false, labels: modes });
  return field({
    label: "External afferents for the first layer", control, key: NET + ".transfer_params",
    help: "This choice applies to every first-layer population. Biological models receiving both signs require distinct afferents. Matched afferents carry the same features; separate groups partition the channels. ON/OFF encoding is set independently in the network input options.",
  });
}

function targetEdges(node) {
  return state.preview.edges.filter(e => e.layer === node.layer)
    .map((edge, index) => ({ edge, index })).filter(({ edge }) => edge.target_id === node.id);
}

function levelText(edge) {
  if (edge.target_depths == null) return "all levels";
  return edge.target_depths.map(depth => depth === 0 ? "soma" : "depth " + depth).join(", ");
}

export function externalControls(node) {
  const connectionsPath = NET + ".layers.0.connections";
  const incoming = targetEdges(node);
  const connections = incoming.filter(({ edge }) => external.has(edge.source_id));
  const controls = [];
  for (const [label, pathway, source] of [["E", "ff_excitatory", "input_e"], ["I", "ff_inhibitory", "input_i"]]) {
    const routes = connections.filter(({ edge }) => edge.pathway === pathway);
    if (!routes.length) {
      controls.push(field({
        label: "Direct external " + label,
        control: switchInput(false, () => edit(cfg => {
          const layer = getPath(cfg, NET).layers[0];
          (layer.connections ||= []).push({ source, target: node.name });
        }, { path: connectionsPath, structural: true }), "Direct external " + label),
        help: "Enable to connect " + source + " to " + node.name + ".",
      }));
    }
    for (const {edge, index} of routes) {
      const title = "Direct external " + label + (routes.length > 1 ? " · route " + (index + 1) : "");
      const onlyExcitation = edge.enabled !== false && pathway === "ff_excitatory" &&
        !incoming.some(route => route.index !== index && route.edge.enabled !== false && route.edge.pathway === pathway);
      const toggle = switchInput(edge.enabled !== false, value => edit(cfg => {
        getPath(cfg, connectionsPath)[index].enabled = value;
      }, { path: connectionsPath, structural: true }), title);
      toggle.disabled = onlyExcitation;
      const placement = edge.source + " → " + node.name + " · " + levelText(edge);
      controls.push(field({
        label: title, key: connectionsPath + "." + index + ".enabled", control: toggle,
        help: onlyExcitation ? placement + ". Connect another same-step E source before turning this input off." : placement,
      }));
    }
  }
  controls.push(inputIdentity());
  const active = new Set(state.preview.edges.filter(e => e.layer === 0 && e.enabled !== false).map(e => e.source_id));
  if (state.preview.biological_neuron && afferentMode(getPath(state.config, NET)) === "shared" &&
      active.has("input_i") && (active.has("input") || active.has("input_e"))) {
    controls.push(h("p", { class: "notice warn compact", role: "status", text: "Both signs currently read shared input channels. Choose matched afferents or separate channel groups to build this biological model." }));
  }
  return controls;
}

function receivingLevels(node, edge) {
  if (edge.enabled === false) return levelText(edge) + " (off)";
  const depths = node.received_per_level[edge.pathway].flatMap((count, depth) =>
    count && (edge.target_depths == null || edge.target_depths.includes(depth)) ? [depth] : []);
  if (!depths.length) return "no contacts";
  const labels = depths[0] === 0 ? ["soma"] : [];
  const branches = depths.filter(depth => depth > 0);
  if (branches.length === 1) labels.push("depth " + branches[0]);
  if (branches.length > 1) {
    const continuous = branches.at(-1) - branches[0] + 1 === branches.length;
    labels.push("depths " + (continuous ? branches[0] + "–" + branches.at(-1) : branches.join(", ")));
  }
  return labels.join(" + ");
}

export function sourcesTable(node) {
  const edges = targetEdges(node);
  const rows = [];
  for (const [path, sign] of [["ff_excitatory", "E"], ["ff_inhibitory", "I"], ["rec_excitatory", "E · delayed"], ["rec_inhibitory", "I · delayed"]]) {
    const routes = edges.filter(({ edge }) => edge.pathway === path);
    if (path.startsWith("rec_") && !routes.length && !node.configured_per_level[path].some(Boolean)) continue;
    const total = node.received_per_level[path].reduce((sum, n, depth) => sum + n * node.levels_including_soma[depth], 0);
    const links = routes.map(({edge, index}) => {
      const origin = external.has(edge.source_id) ? "External " + edge.source : edge.source_id;
      return h("div", {}, [h("button", { type: "button", class: "linkish",
        onclick: () => select({ kind: "connection", layer: edge.layer, index }), text: origin,
      }), " · " + receivingLevels(node, edge)]);
    });
    rows.push(h("tr", {}, [h("th", { scope: "row", text: sign }), h("td", {}, links.length ? links : ["No source"]), h("td", { text: fmt(total) })]));
  }
  return h("section", { "aria-label": "Synapse sources per neuron" }, [
    h("h4", { text: "Synapse sources per neuron" }),
    h("table", { class: "levels sources" }, [h("thead", {}, [h("tr", {}, ["Synapses", "Eligible sources", "Contacts"].map(text => h("th", { scope: "col", text })))]), h("tbody", {}, rows)]),
    h("p", { class: "help", text: "Each total is the contact budget across the neuron's compartments. Sources on the same pathway share that budget. The level table gives contacts per compartment." }),
  ]);
}

// In structured feedforward families the architecture fixes the routes.
// Counts stay in the existing typed Connectivity form.
export function structuredInputGuide(preview) {
  const core = preview.resolved_config.model.core;
  const families = ["einet", "dendritic_shunting", "dendritic_additive", "dendritic_normalized_additive",
    "flat_shunting", "flat_additive", "flat_normalized_additive"];
  if (preview.recurrent || !families.includes(preview.family)) return null;
  const architecture = core.architecture;
  const hasLocalI = Boolean(architecture.inhibitory_layer_sizes?.[0]);
  const denseI = hasLocalI && architecture.inhibitory_network_type === "mlp";
  const rows = [];
  const add = (cells, sign, source, key) => {
    const count = core.connectivity[key + "_synapses_per_branch_per_layer"]?.[0] ?? 0;
    rows.push([cells, sign, source, String(count)]);
  };
  add("E cells", "E", "External E stream", "ee");
  add("E cells", "I", hasLocalI ? "Local I cells" : "External I stream", "ie");
  if (hasLocalI && !denseI) {
    add("I cells", "E", "External E stream", "ei");
    add("I cells", "I", "External I stream", "ii");
  }
  return h("section", { "aria-label": "First-layer feedforward sources" }, [
    h("h4", { text: "First-layer feedforward sources" }),
    h("table", { class: "levels" }, [h("thead", {}, [h("tr", {}, ["Receiver", "Synapses", "Source", "Per branch"].map(text => h("th", { scope: "col", text })))]),
      h("tbody", {}, rows.map(row => h("tr", {}, row.map(text => h("td", { text })))))]),
    denseI ? h("p", { class: "help", text: "The local inhibitory population is a dense MLP driven by the excitatory input." }) : null,
    h("p", { class: "help", text: "Edit per-branch counts under Connectivity. A zero count omits that pathway. Architecture sets the local inhibitory population size; Transfer sets how the external channels are supplied." }),
  ]);
}

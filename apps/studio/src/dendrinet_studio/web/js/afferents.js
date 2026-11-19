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

function afferentMode(net) {
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
  return [...controls, inputIdentity()];
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
      }), " · " + levelText(edge) + (edge.enabled === false ? " (off)" : "")]);
    });
    rows.push(h("tr", {}, [h("th", { scope: "row", text: sign }), h("td", {}, links.length ? links : ["No source"]), h("td", { text: fmt(total) })]));
  }
  return h("section", { "aria-label": "Synapse sources per neuron" }, [
    h("h4", { text: "Synapse sources per neuron" }),
    h("table", { class: "levels sources" }, [h("thead", {}, [h("tr", {}, ["Synapses", "Eligible sources", "Contacts"].map(text => h("th", { scope: "col", text })))]), h("tbody", {}, rows)]),
    h("p", { class: "help", text: "Each total is the contact budget across the neuron's compartments. Sources on the same pathway share that budget. The level table gives contacts per compartment." }),
  ]);
}

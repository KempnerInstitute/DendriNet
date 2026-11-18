DendriNet Studio
================

Studio is an optional local browser interface for building population-network
configurations. It edits ordinary YAML and uses the library's existing loader,
constructors and training command. It does not implement a second model or
training loop.

From a checkout containing Studio:

.. code-block:: bash

   pip install -e . -e apps/studio
   dendrinet-studio --workspace /path/to/runs

Open the URL printed by the launcher. Start with ``synthetic_population`` for
a small, download-free example. Select populations to set neuron counts,
branching factors, somatic synapses, E/I contact counts, integration, leak and
sparsity. Configure connections between populations and inspect the network
and a representative neuron's morphology. Shape and line style distinguish
excitation, inhibition and training curves independently of color.

The Configuration tab retains advanced options that have no visual control.
YAML comments and formatting are not preserved. Imports must be standalone
experiments: resolve composed configs and sweeps with core tools first.

``Check model`` constructs the network in a separate CPU process and reports
the library's resource counts. ``Train`` uses the normal training CLI and
records the config, command and package identity. The results tab displays
actual worker logs and training curves. Preview counts describe configured
morphology; they are not a trained connectivity map. Constructor validation
and data-dependent calibration happen separately from the fast preview.

Studio binds to the local loopback interface and is not a hosted multi-user
service. On a compute node, use ``--no-browser`` and SSH port forwarding.
Construction and training are refused on login nodes. One worker runs at a
time; distributed training and cluster submission remain CLI workflows.

The first version supports ``population_network``. Other core families,
checkpoint activity playback and arbitrary cable geometry are not graphical
features yet. The app lives in ``apps/studio/``, outside the installed core
package. Its README describes tests and current scope.

Initialization Comparison Policy
================================

This page defines the comparison contract for additive and shunting DendriNet
experiments. It applies to benchmark, neural-data, and morphology studies. The
goal is to keep three different scientific questions from being conflated.

Comparison estimands
--------------------

Best attainable trained system
   Select one complete recipe per mechanism using the same validation
   information, search space, compute budget, seeds, and endpoint hierarchy.
   Mechanism-specific initializers are allowed. The result compares trained
   systems under the tested budget; it is not a pure integration-rule effect
   and does not prove global architecture capacity.

Same-policy robustness control
   Give both mechanisms the same named initializer and optimizer. This tests
   sensitivity to recipe symmetry, but the numerical gate parameters and raw
   voltage scales need not be identical.

Rule-only causal control
   Hold the serialized epoch-zero state, connectivity, data order, and optimizer
   fixed while changing only the forward integration equation. This supports a
   narrow causal statement at that operating point, not a best-system claim.

Initializer definitions
-----------------------

``analytical`` derives weights and gate parameters from contact counts,
morphology, weight-transform statistics, and expected conductances.

``occupancy_quantile`` is already a two-stage analytical-weight/empirical-gate
initializer. The code first applies analytical synaptic initialization. It then
passes unlabeled training inputs through the realized weights and morphology and
maps observed voltage quantiles to fixed gate-output targets. It uses no labels,
validation/test endpoints, or fitted mixing coefficient.

The diagnostic component policies are:

* ``analytical_slope_occupancy_center``: ``(m_a,b_o)``;
* ``occupancy_slope_analytical_center``: ``(m_o,b_a)``.

Together with analytical ``(m_a,b_a)`` and occupancy ``(m_o,b_o)``, they form a
2-by-2 factorial that attributes sensitivity to voltage centering, voltage
scale, or their interaction. These policies are diagnostics, not additional
default initializers. A continuous analytical--occupancy interpolation was
removed because it introduced a tunable mixing parameter without a distinct
scientific estimand.

Selection and confirmation protocol
-----------------------------------

#. Declare the initializer and optimizer grid before reading test endpoints.
#. Give each mechanism the identical validation information and number of
   trials. Pair seeds and all structural/data conditions.
#. Select the mean validation endpoint across declared conditions; never select
   a seed or condition on test performance.
#. Record the pre-calibration gate, calibration inputs and convergence, applied
   post-calibration gate, realized contacts, and source/config hashes.
#. Freeze the selected recipe and run fresh confirmation seeds.
#. Evaluate the held-out test, gain, corruption/noise, and information endpoints
   once. Report the full same-policy and rule-only controls separately.

Current project decisions (2026-07-15)
--------------------------------------

Mouse V1 NeurIPS project
   An 864-run validation-only sweep independently selected analytical
   initialization and paired dendritic/reactivation learning rate ``3e-4`` for
   both mechanisms. At that learning rate, analytical minus occupancy validation
   accuracy was +0.186 percentage points for additive (paired 95% CI
   [+0.071,+0.302]) and +0.187 points for shunting ([+0.056,+0.318]). The
   provisional primary recipe is therefore analytical-both at ``3e-4``. An
   occupancy-both rerun remains a sensitivity control. Final claims require the
   guarded fresh-seed confirmation because legacy arrays did not persist the
   post-calibration gate state.

Empirical CIFAR project
   Under the locked CIFAR optimizer and input statistics, the validated primary
   recipes remain analytical shunting and occupancy additive. This is compatible
   with the V1 decision: initializer optimality is dataset- and optimizer-
   dependent, so cross-project recipe identity is not a fairness requirement.
   The parameter-free component factorial should diagnose the origin of that
   difference before any new default is promoted.

Reporting language
------------------

Use ``validation-selected attainable performance`` for the first estimand,
``same-policy sensitivity`` for the second, and ``rule-only control`` for the
third. Avoid ``maximum capacity`` unless optimization over the declared function
class is actually exhaustive. Avoid describing an initializer-dependent trained
gap as a passive shunting advantage.

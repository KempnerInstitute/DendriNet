# Empirical figure index

| Main figure | Question | Accompanying material |
| --- | --- | --- |
| 1 | Which features feed individual digit-selective neurons? | MNIST pair; S1 |
| 2 | How do E/I allocation, contact budget and routing affect accuracy? | S2–S7, S15 |
| 3 | How much does activity at each dendritic depth influence the soma and classification? | Primary CIFAR-10 pair; numerical replay of 3D; S8–S12 |
| 4 | How do depth and integration affect sensitivity to external noise and gain? | S13–S17 |
| 5 | What changes when the same contacts are pooled in soma-only neurons or fewer compartments? | Contact-budget and pooling comparisons in the manuscript |
| 6 | How do signals and local inhibition contribute across network layers? | Three-layer CIFAR-10 pair; S18–S23 |
| 7 | How do gain sensitivity and cue-specific inhibition affect the answer to the same character? | Contextual pair, training calibration and numerical replay of 7F; S24–S26 |
| 8 | How do dendritic replacements work at AlexNet width? | Historical ImageNet checkpoint cohorts; S27–S28 |
| S29 | How are the examples built and measured? | Tutorial and CIFAR-10 wiring, seed summaries from 3D and 7F, figure renderer |

DendriNet v0.3.0 (commit `8e884ea37f470b33f6557a8c0c0734369d273667`)
generated the native neuronal examples. The AlexNet checkpoints retain their
historical training provenance. Figure 8F and S28 used inference code at
`dc4389d9bfb94a0b6871141d8084ee1dc6ee0bce`. The archived results reported for
the original contextual implementation have their own source.

The code and tables in this directory reproduce the two identified numerical
panels and the software example. Other rows locate the corresponding
experiments in the manuscript. Re-running a complete figure also requires its
cohort prescriptions, analysis and numerical inputs.

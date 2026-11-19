# Research projects

This directory is a home for approved study releases. A study can instead
remain in a companion repository linked from DendriNet. Each study has one
canonical source location; avoid two independently edited copies.

Use the following structure for a study added here:

```text
projects/<study>/
    README.md             reproduction instructions and scope
    configs/              base recipes, sweeps, explicit overrides
    analysis/             cohort assembly and statistical comparisons
    figures/              scripts and panel assembly
    tests/                checks specific to the study
    reproducibility/      source pins, environment, artifact hashes and locations
```

Reusable model equations, trainers, controls and measurement primitives belong
in `src/dendritic_modeling/`. Study files import those capabilities. The library
never imports a study, and projects are excluded from its wheel and source
distribution. Large datasets and checkpoints belong in an artifact repository.

Moving a study does not change its generation pins or qualify a newer library.
Record the old and new source identity and compare each affected entry point on
fixed inputs and checkpoints before adopting a new release. Keep generation
and replay versions separate. Publish only approved sources and artifacts;
do not include a study's internal history or a vendored library runtime.

The [empirical study](empirical/README.md) includes source-matched training
configurations and two analyses with small, separately distributable source
tables. For another paper, link its canonical release here whether it lives
under `projects/` or in the authors' repository. Cite a version of DendriNet
and a version of that study's code and data.

For each quantitative figure, identify its cohort, input table, units and
analysis. Include the transformations and statistics needed to recover the
reported numbers. A single current renderer is sufficient where the drawing
encodes a scientific choice such as binning, masking or normalization.
Alternative layouts, drafts, scheduler state and reviewer correspondence
belong in the authors' workspace. Large or licensed inputs can be described
by immutable identifiers, acquisition instructions and access terms.

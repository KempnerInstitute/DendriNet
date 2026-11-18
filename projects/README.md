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

No existing paper has been moved or re-pinned by creating this directory.

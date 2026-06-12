# Release Packages

This directory is reserved for sanitized public reproducibility packages.

No active draft, private experiment, cluster-specific config, raw result,
checkpoint, Slurm log, or personal path should live here. While work is in
development, keep working material in the private `drafts/` tree or external
storage.

## Current Policy

Project folders are ignored by default. This keeps the main DendriNet codebase
clean and prevents accidental commits of private or machine-specific material.

When a reproducibility package is ready for public release, create one
sanitized package here with only releaseable assets:

```text
projects/<project_id_slug>/
  README.md
  REPRODUCIBILITY.md
  ARTIFACTS.md
  release_manifest.yaml
  configs/
  scripts/
```

Optional small folders such as `figures/` or `results/` are acceptable only when
they contain compact, sanitized, release-facing assets. Large outputs should be
hosted externally, for example on Hugging Face, Zenodo, OSF, or institutional
storage.

## Release Checklist

Before unignoring and committing a project folder here:

1. Remove or replace all private absolute paths, usernames, accounts, and
   cluster partitions.
2. Keep raw checkpoints, full run directories, logs, and large arrays outside
   git.
3. Move reusable implementation into `src/dendritic_modeling/` with tests.
4. Include only canonical configs and compact summary files required for release
   reproduction.
5. Add artifact preparation and verification scripts if external artifacts are
   released.
6. Run a private-path scan before commit.

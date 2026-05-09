# Future Release Package Template

Use this template only when a sanitized public reproducibility package is ready
to live under `projects/<project_id_slug>/`.

```text
projects/<project_id_slug>/
  README.md
  REPRODUCIBILITY.md
  ARTIFACTS.md
  release_manifest.yaml
  configs/
  scripts/
```

## README.md

```markdown
# Project Title

Short description of the release package.

## Scope

What this package reproduces, what it does not include, and how it relates to
the reusable `dendritic_modeling` package.

## Code Entry Points

- Train:
- Evaluate:
- Collect results:
- Generate figures:

## External Artifacts

List public artifact locations for checkpoints, raw runs, or large derived
outputs. Do not store large artifacts in git.
```

## REPRODUCIBILITY.md

```markdown
# Reproducibility

## Environment

Python, PyTorch, CUDA, package, and cluster assumptions.

## Minimal Reproduction

Commands that rebuild release figures or tables from tracked summaries.

## Full Reproduction

Commands that rerun training/evaluation from canonical configs.

## Known Limitations

Expected runtime, incomplete seeds, nondeterminism, and external dependencies.
```

## ARTIFACTS.md

```markdown
# Artifact Plan

State what belongs in GitHub, what belongs in external storage, and what must
never be released.
```

## release_manifest.yaml

```yaml
project: project_id_slug
code_package: dendritic_modeling
public_release:
  code:
    include:
      - src/dendritic_modeling
      - configs required by the release
      - projects/project_id_slug
    exclude:
      - drafts
      - logs
      - checkpoints
      - raw results
      - private paths
  artifacts:
    location: external artifact repository
    include:
      - compact result summaries
      - generated release figures, if needed
      - checksums and metadata
    exclude:
      - raw logs
      - private cluster paths
      - full checkpoints unless intentionally released
```

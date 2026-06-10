# Configuration Layout

This directory is the public configuration surface for the codebase.

- `config_exp.yaml`: complete single-GPU reference config with documented fields.
- `config_exp_fsdp.yaml`: complete FSDP / multi-GPU reference config.
- `sweep_config.yaml`: maintained generic sweep template.
- `examples/`: small, documented examples for common model families and features.
- `vision/`: AlexNet / vision-backbone replacement examples.

Operational benchmark packs live with their job owners. Transformer replacement
benchmark configs are under `jobs/transformer_replacement_benchmarks/configs/`.
Project- or draft-specific sweep packs should live under the relevant
`drafts/*/configs/` tree or another project-owned directory, not here.

Ignored local/private config archives may live under `.vscode/archived_configs/`
or `projects/private_configs/`.

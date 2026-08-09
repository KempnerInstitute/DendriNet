# Vision replacement and full-model configurations

The configurations in this directory use the ordinary training and sweep
entry points. They do not require paper-local analysis code.

## AlexNet replacement sequence

1. `alexnet_conv1_to_conv4_replacement_screen.yaml` and
   `alexnet_conv5_replacement_benchmark.yaml` test spatial replacements while
   retaining an exact pretrained suffix.
2. `alexnet_fc6_replacement_benchmark.yaml` and
   `alexnet_fc6fc7_replacement_benchmark.yaml` test classifier replacements.
3. `dendritic_alexnet_imagenet_boundary_capacity_pilot.yaml` fits conv1--fc7
   against exact teacher inputs, and
   `dendritic_alexnet_imagenet_class_output_capacity_pilot.yaml` fits the
   strict one-soma-per-class fc8 boundary.
4. `dendritic_alexnet_imagenet_output_organization_pilot.yaml` compares one
   soma per channel with fixed-mean and learned-positive within-channel soma
   populations.

`vision.output_organization.mode` accepts `one_to_one`, `fixed_mean`, or
`learned_positive`. `spatial_somas_per_channel` is a scalar or five-element
list, and `classifier_somas_per_unit` is a scalar or two-element list for fc6
and fc7. The learned reduction is a separate positive sum-to-one weighting
within each declared channel or hidden-unit identity; it cannot mix identities.
The final classifier always remains one excitatory soma per class.
For its teacher-forced capacity fit, `class_output_target` uses
`softmax_equivalent_nonnegative`: the minimum teacher logit is subtracted from
all classes within each example. This makes the target nonnegative without
changing its softmax probabilities and does not add a learned decoder.

## Hash-bound boundary assembly

After each boundary has been evaluated, assemble the selected checkpoints with
the package CLI:

```bash
dendritic-assemble-vision-boundaries --manifest assembly.yaml
```

The manifest uses the following schema. Every checkpoint must have the same
state-dict tensor inventory; each digest is verified before any output is
written.

```yaml
schema_version: dendritic_vision_checkpoint_assembly_v1
output_checkpoint: /absolute/path/assembled.pt
sources:
  conv1: {path: /absolute/path/conv1.pt, sha256: <64 hex characters>}
  conv2: {path: /absolute/path/conv2.pt, sha256: <64 hex characters>}
  conv3: {path: /absolute/path/conv3.pt, sha256: <64 hex characters>}
  conv4: {path: /absolute/path/conv4.pt, sha256: <64 hex characters>}
  conv5: {path: /absolute/path/conv5.pt, sha256: <64 hex characters>}
  fc6: {path: /absolute/path/fc6.pt, sha256: <64 hex characters>}
  fc7: {path: /absolute/path/fc7.pt, sha256: <64 hex characters>}
  fc8: {path: /absolute/path/fc8.pt, sha256: <64 hex characters>}
  remainder: {path: /absolute/path/reference.pt, sha256: <64 hex characters>}
metadata:
  study: declared_study_name
```

The command writes both the assembled checkpoint and a provenance JSON record.
Use the emitted file digest to bind subsequent training:

```yaml
model:
  initial_checkpoint: /absolute/path/assembled.pt
  initial_checkpoint_sha256: <digest emitted by the assembler>
  initial_checkpoint_strict: true
```

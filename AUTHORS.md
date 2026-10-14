# Authors

DendriNet (`dendritic_modeling`) is developed at the Kempner Institute for the
Study of Natural and Artificial Intelligence at Harvard University.

## Contributors

- Houman Safaai
- Maceo Richards
- Naeem Khoshnevis
- Ryan Whalen
- Ilenna Jones

## Credits for specific components

- Indexed-synapse GPU kernels: Ryan Whalen profiled the indexed-gather runtime
  and contributed the CUDA/Triton gather kernels, including the
  transposed-coalesced forward, weight-gradient and atomic input-gradient
  kernels (`triton_transposed`). Houman Safaai integrated them into the
  configuration, dispatch, model and test paths.
- Fixed-fan-in ELL kernels (`dendritic_modeling.kernels.triton_ell`): Houman
  Safaai developed this separate feature-major family of shunting and gated-GLU
  operators.

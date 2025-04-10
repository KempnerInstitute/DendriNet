# Gradient Hook Functionality Fix

This branch (`iss192_hook_hs`) fixes the gradient hook functionality that was working in `iss157_sweep_ei_MR` but not in `iss190_debug_new_hs`.

## Key Issues Fixed

1. **In `gradient_scaling.py`**:
   - Uncommented the code that clears existing hooks before registering new ones
   - Re-enabled safe hook wrapper functions
   - Added proper debug prints to trace hook registration and execution
   - Fixed gradient checking in the safe hook decorator
   - Added tensor-based hooks in addition to module-based hooks
   - Implemented robust fallback mechanisms when scale vectors aren't initialized

2. **In `dendrinet.py`**:
   - Fixed `BlockLinear.forward()` and `TopKLinear.forward()` methods:
     - Uncommented code that ensures input tensors require gradients
     - Made sure `retain_grad()` is called to ensure input hooks get triggered
     - Added debugging prints to verify gradient status
     - Added direct tensor hooks as an additional safety measure

## How to Test the Fix

1. Run the test script:
   ```
   python test_hooks.py
   ```

   This script creates simple models with hook registrations and performs forward/backward passes. If hooks are working correctly, you'll see debug messages from the hooks during backward pass.

2. Verify hooks with your own training code:
   ```python
   from src.dendritic_modeling.dendrinet import DendriNet

   # Create model with hook strategies
   model = DendriNet(
       n_soma=10,
       branch_factors=[2],
       excitatory_input_dim=20,
       excitatory_synapses_per_branch=5,
       topk_strategy="conductance_dynamic",  # Enable topk hooks
       blocklinear_strategy="conductance_dynamic",  # Enable blocklinear hooks
   )

   # Train as usual - hooks should be registered and triggered during backprop
   ```

## Explanation of Changes

1. **Dual Hook System**:
   - Module-based hooks using `register_full_backward_hook`
   - Tensor-based hooks using direct `tensor.register_hook`
   - Both systems work in parallel for maximum reliability

2. **Layer Forward Methods**:
   - Input tensors are properly set up with `requires_grad=True`
   - `retain_grad()` ensures non-leaf tensors also trigger hooks
   - Direct tensor hooks added as a failsafe mechanism

3. **Hook Registration**:
   - Existing hooks are cleared before registering new ones
   - Safe hook wrappers handle edge cases
   - Both module hooks and parameter hooks are properly registered
   - Custom forward method is injected for tensor-based hooks

4. **Fallback Mechanisms**:
   - Added robust fallbacks when scale vectors aren't initialized
   - Created default scale vectors when needed
   - Improved error handling with detailed error messages and stack traces

5. **Debug Enhancements**:
   - Clear debug messages for hook registration and execution
   - Shape information for inputs and gradients
   - Better error reporting with stack traces

## Common Issues

If hooks are still not being called, check the following:

1. Verify that the model is in training mode (`model.train()`)
2. Make sure the input tensors have `requires_grad=True`
3. Confirm that loss.backward() is being called
4. Check that the gradient strategies are set to a non-"none" value
5. Double-check that layer.register_hook() calls aren't being overridden elsewhere

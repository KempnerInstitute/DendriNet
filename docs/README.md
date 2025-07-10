# Documentation

This directory contains the documentation for the Dendritic Modeling Framework.

## Structure

- `index.rst` - Main documentation homepage
- `installation.rst` - Installation instructions
- `tutorials.rst` - Hands-on tutorials
- `api_reference.rst` - API documentation
- `theory.rst` - Scientific background
- `examples.rst` - Practical examples
- `contributing.rst` - Contribution guidelines
- `available_components.md` - List of all implemented components

## Figures

All figures are stored in the `figures/` directory:
- `figures/architecture/` - Network architecture diagrams
- `figures/README.md` - Information about the figures

## Building Documentation

To build the documentation locally:

```bash
cd docs
make html
```

The built documentation will be in `_build/html/`.

## Notes

- The documentation has been updated to accurately reflect what's implemented in the codebase
- Components that were mentioned but not implemented (like DendriticRNN, DendriticTransformer, etc.) have been removed
- All examples now use actual components from the codebase 
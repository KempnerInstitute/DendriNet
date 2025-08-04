Contributing Guide
==================

Development Setup
-----------------

1. Fork and clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/dendritic-modeling.git
      cd dendritic-modeling

2. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate

3. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

4. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Development Workflow
-------------------

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. Make your changes

3. Run tests:

   .. code-block:: bash

      pytest tests/
      pytest --cov=dendritic_modeling tests/

4. Check code style:

   .. code-block:: bash

      flake8 src/
      black --check src/

5. Commit and push:

   .. code-block:: bash

      git add .
      git commit -m "feat: add new feature"
      git push origin feature/your-feature-name

Code Standards
--------------

- Follow PEP 8 with 88 character line length
- Use type hints for all public functions
- Write docstrings in NumPy style
- Include tests for all new code
- Ensure GPU compatibility where applicable

Example function:

.. code-block:: python

   def compute_shunting_inhibition(
       excitation: torch.Tensor,
       inhibition: torch.Tensor,
       epsilon: float = 1e-6
   ) -> torch.Tensor:
       """
       Compute shunting inhibition response.
       
       Parameters
       ----------
       excitation : torch.Tensor
           Excitatory input, shape (batch_size, n_neurons)
       inhibition : torch.Tensor
           Inhibitory input, shape (batch_size, n_neurons)
       epsilon : float, default=1e-6
           Small constant for numerical stability
           
       Returns
       -------
       torch.Tensor
           Normalized response, shape (batch_size, n_neurons)
       """
       return excitation / (excitation + inhibition + epsilon)

Testing Requirements
-------------------

All new code must include tests:

.. code-block:: python

   def test_shunting_inhibition_basic():
       """Test basic shunting inhibition computation."""
       exc = torch.ones(10, 20)
       inh = torch.ones(10, 20)
       
       result = compute_shunting_inhibition(exc, inh)
       
       assert result.shape == (10, 20)
       assert torch.allclose(result, torch.full((10, 20), 0.5), atol=1e-5)

Pull Request Checklist
---------------------

Before submitting:

- Tests pass locally
- Code follows style guidelines
- Documentation is updated
- No merge conflicts
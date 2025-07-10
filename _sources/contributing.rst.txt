Contributing Guide
==================

Thank you for your interest in contributing to the Dendritic Modeling Framework! This guide will help you get started with contributing code, documentation, examples, or ideas.

Code of Conduct
--------------

We are committed to providing a welcoming and inclusive environment. All contributors are expected to adhere to our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

Getting Started
--------------

Setting Up Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork and clone the repository**:

   .. code-block:: bash

      git clone https://github.com/yourusername/dendritic-modeling.git
      cd dendritic-modeling

2. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install in development mode**:

   .. code-block:: bash

      pip install -e ".[dev]"

4. **Install pre-commit hooks**:

   .. code-block:: bash

      pre-commit install

Development Workflow
~~~~~~~~~~~~~~~~~~~

1. **Create a feature branch**:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

2. **Make your changes**

3. **Run tests**:

   .. code-block:: bash

      pytest tests/
      pytest --cov=dendritic_modeling tests/  # With coverage

4. **Check code style**:

   .. code-block:: bash

      flake8 src/
      black --check src/
      mypy src/

5. **Commit changes**:

   .. code-block:: bash

      git add .
      git commit -m "feat: add new feature"

6. **Push and create pull request**:

   .. code-block:: bash

      git push origin feature/your-feature-name

Contributing Code
----------------

Architecture Guidelines
~~~~~~~~~~~~~~~~~~~~~~

When adding new features, follow these architectural principles:

1. **Biological Inspiration**: Ensure new features align with neuroscience principles
2. **Modularity**: Create reusable components that integrate well
3. **Performance**: Consider computational efficiency
4. **Documentation**: All public APIs must be documented

Code Style
~~~~~~~~~

We follow PEP 8 with these additions:

- Line length: 88 characters (Black default)
- Use type hints for all public functions
- Docstrings in NumPy style

Example:

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
           
       Examples
       --------
       >>> exc = torch.rand(32, 100)
       >>> inh = torch.rand(32, 100)
       >>> response = compute_shunting_inhibition(exc, inh)
       """
       return excitation / (excitation + inhibition + epsilon)

Testing Requirements
~~~~~~~~~~~~~~~~~~~

All new code must include tests:

.. code-block:: python

   # tests/test_shunting.py
   import pytest
   import torch
   from dendritic_modeling import compute_shunting_inhibition
   
   def test_shunting_inhibition_basic():
       """Test basic shunting inhibition computation."""
       exc = torch.ones(10, 20)
       inh = torch.ones(10, 20)
       
       result = compute_shunting_inhibition(exc, inh)
       
       assert result.shape == (10, 20)
       assert torch.allclose(result, torch.full((10, 20), 0.5), atol=1e-5)
   
   def test_shunting_inhibition_edge_cases():
       """Test edge cases."""
       # Zero inhibition
       exc = torch.ones(5, 10)
       inh = torch.zeros(5, 10)
       result = compute_shunting_inhibition(exc, inh)
       assert torch.allclose(result, torch.ones(5, 10), atol=1e-5)
       
       # Zero excitation
       exc = torch.zeros(5, 10)
       inh = torch.ones(5, 10)
       result = compute_shunting_inhibition(exc, inh)
       assert torch.allclose(result, torch.zeros(5, 10), atol=1e-5)

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~

When implementing computationally intensive features:

1. **Profile first**: Use PyTorch profiler to identify bottlenecks
2. **Optimize hotspots**: Focus on frequently called functions
3. **Consider GPU**: Ensure operations are GPU-friendly
4. **Benchmark**: Compare against baseline implementations

Contributing Documentation
-------------------------

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~

- Use clear, concise language
- Include code examples
- Add figures where helpful
- Cross-reference related topics

Adding Tutorials
~~~~~~~~~~~~~~~

Create tutorials that:

1. Start with motivation
2. Build incrementally
3. Include complete, runnable code
4. Explain key concepts
5. Show results/visualizations

Example structure:

.. code-block:: rst

   Tutorial: Building a Custom Dendritic Layer
   ===========================================
   
   In this tutorial, we'll create a custom dendritic layer with 
   specialized plasticity rules.
   
   Motivation
   ----------
   
   Biological neurons exhibit diverse plasticity mechanisms...
   
   Implementation
   -------------
   
   Let's start by defining our custom layer:
   
   .. code-block:: python
   
      class CustomDendriticLayer(nn.Module):
          ...
   
   Key Concepts
   -----------
   
   This implementation incorporates three key ideas:
   
   1. **Heterosynaptic plasticity**: ...
   2. **Dendritic spikes**: ...
   3. **Branch-specific learning**: ...

Contributing Examples
--------------------

Example Requirements
~~~~~~~~~~~~~~~~~~~

Each example should include:

1. **README.md**: Clear description and instructions
2. **requirements.txt**: Specific dependencies
3. **config.yaml**: Configuration file
4. **main.py**: Main script
5. **visualize.py**: Visualization code
6. **results/**: Sample outputs

Template structure:

.. code-block:: text

   examples/your_example/
   ├── README.md
   ├── requirements.txt
   ├── config.yaml
   ├── main.py
   ├── utils.py
   ├── visualize.py
   └── results/
       ├── figures/
       └── checkpoints/

Research Contributions
---------------------

Publishing Research
~~~~~~~~~~~~~~~~~~

If you use the framework in research:

1. **Cite the framework**: Use our citation format
2. **Share your code**: Create an example
3. **Report results**: Add to our benchmark table
4. **Describe methods**: Document novel techniques

Proposing New Features
~~~~~~~~~~~~~~~~~~~~~

For significant new features:

1. **Open an issue**: Discuss the proposal
2. **Write a design doc**: Outline the approach
3. **Implement prototype**: Show feasibility
4. **Gather feedback**: Iterate on design

Review Process
-------------

Pull Request Checklist
~~~~~~~~~~~~~~~~~~~~~

Before submitting:

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Changelog entry added
- [ ] No merge conflicts

Review Timeline
~~~~~~~~~~~~~~

- **Initial review**: Within 3-5 days
- **Feedback incorporation**: As needed
- **Final approval**: 1-2 days after changes

What Reviewers Look For
~~~~~~~~~~~~~~~~~~~~~~

1. **Correctness**: Does it work as intended?
2. **Design**: Is it well-architected?
3. **Testing**: Is it properly tested?
4. **Documentation**: Is it well-documented?
5. **Performance**: Is it efficient?

Community
---------

Getting Help
~~~~~~~~~~~

- **Discord**: Join our community server
- **GitHub Discussions**: Ask questions
- **Issues**: Report bugs or request features
- **Twitter**: Follow @dendritic_ml

Regular Events
~~~~~~~~~~~~~

- **Weekly Office Hours**: Thursdays 4pm EST
- **Monthly Journal Club**: First Tuesday
- **Hackathons**: Quarterly

Recognition
~~~~~~~~~~

We recognize contributors through:

- Contributors list in README
- Author credits in CHANGELOG
- Highlight in release notes
- Community spotlight blog posts

Thank You!
----------

Your contributions make this project better for everyone. Whether you're fixing a typo, adding a feature, or sharing your research, every contribution matters.

Welcome to the Dendritic Modeling community! 🧠🌳 
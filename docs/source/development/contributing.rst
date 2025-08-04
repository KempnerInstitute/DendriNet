Contributing to Dendritic Modeling Framework
==========================================

We welcome contributions to the Dendritic Modeling Framework! This guide will help you set up your development environment, understand our development workflow, and make meaningful contributions.

Quick Start for Contributors
---------------------------

.. code-block:: bash

    # 1. Fork the repository on GitHub
    # 2. Clone your fork
    git clone https://github.com/your-username/dendritic-modeling.git
    cd dendritic-modeling

    # 3. Set up development environment
    conda env create -f environment.yml
    conda activate dendritic_env
    pip install -e ".[dev]"

    # 4. Set up pre-commit hooks
    pre-commit install

    # 5. Create a feature branch
    git checkout -b feature/amazing-feature

    # 6. Make your changes, add tests, commit
    # 7. Push and create a pull request

Development Environment Setup
---------------------------

Installation for Development
~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose one of these methods:

**Option 1: Full Conda Environment (Recommended)**

.. code-block:: bash

    # Creates environment with all dependencies including testing tools
    conda env create -f environment.yml
    conda activate dendritic_env
    pip install -e ".[dev]"

**Option 2: pip with Virtual Environment**

.. code-block:: bash

    python -m venv dendritic_env
    source dendritic_env/bin/activate  # Windows: dendritic_env\Scripts\activate
    pip install -r requirements.txt
    pip install -e .

**Option 3: Install Specific Dependency Groups**

.. code-block:: bash

    # Core + testing only
    pip install -e ".[test]"

    # Core + code quality tools
    pip install -e ".[lint]"

    # Everything for full development
    pip install -e ".[all]"

Setting Up Pre-commit Hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-commit hooks ensure code quality before each commit:

.. code-block:: bash

    # Install pre-commit hooks
    pre-commit install

    # Test the setup
    pre-commit run --all-files

    # Update hooks (occasionally)
    pre-commit autoupdate

Verification
~~~~~~~~~~

Verify your development setup:

.. code-block:: bash

    # Test imports
    python -c "import dendritic_modeling; print('Package working')"

    # Run tests
    pytest tests/ -v

    # Check code quality
    pre-commit run --all-files

    # Run comprehensive checks (same as CI)
    python tests/test_runner.py --mode full

Development Workflow
------------------

Creating a Feature Branch
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Update your main branch
    git checkout main
    git pull upstream main

    # Create feature branch
    git checkout -b feature/descriptive-name

    # Or for bug fixes
    git checkout -b fix/issue-description

Making Changes
~~~~~~~~~~~~

1. **Write your code** following our coding standards
2. **Add tests** for new functionality
3. **Update documentation** if needed
4. **Run tests locally** to ensure everything works

.. code-block:: bash

    # Run tests frequently during development
    pytest tests/test_specific.py -v

    # Check code quality
    ruff check src/ tests/
    black src/ tests/
    isort src/ tests/

    # Run comprehensive test suite
    python tests/test_runner.py

Committing Changes
~~~~~~~~~~~~~~~~

Our pre-commit hooks will automatically run when you commit:

.. code-block:: bash

    # Stage your changes
    git add .

    # Commit (pre-commit hooks run automatically)
    git commit -m "feat: add new dendritic layer functionality

    - Implement new TopK strategy for branch selection
    - Add comprehensive tests for new functionality
    - Update documentation with usage examples
    
    Fixes #123"

**Commit Message Format**

We follow conventional commits:

- ``feat:`` New features
- ``fix:`` Bug fixes
- ``docs:`` Documentation changes
- ``test:`` Adding or modifying tests
- ``refactor:`` Code refactoring
- ``perf:`` Performance improvements
- ``ci:`` CI/CD changes

Creating a Pull Request
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Push your branch
    git push origin feature/your-feature-name

    # Create PR on GitHub with our template

**Pull Request Checklist**

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated if needed
- [ ] Commit messages are clear
- [ ] PR description explains changes

Testing Infrastructure
--------------------

We have a comprehensive testing system with different types of tests:

Test Categories
~~~~~~~~~~~~~

.. code-block:: bash

    # Run all tests
    pytest tests/

    # Run only unit tests (fast)
    pytest -m unit

    # Run only integration tests
    pytest -m integration

    # Skip slow tests
    pytest -m "not slow"

    # Run tests with coverage
    pytest --cov=dendritic_modeling --cov-report=html

Test Organization
~~~~~~~~~~~~~~~

Our test suite includes:

- **Unit Tests** (``tests/test_*.py``): Test individual components
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark critical functions
- **Regression Tests**: Prevent known bugs from returning

**Key test files:**

- ``tests/test_analysis.py``: Analysis and evaluation tools
- ``tests/test_datasets.py``: Dataset loading and processing
- ``tests/test_training_strategies.py``: Training algorithms
- ``tests/test_plotting.py``: Visualization functions
- ``tests/test_config.py``: Configuration system

Writing Tests
~~~~~~~~~~~

When adding new functionality, include tests:

.. code-block:: python

    # tests/test_your_feature.py
    import pytest
    from dendritic_modeling.your_module import YourClass

    class TestYourClass:
        """Test suite for YourClass functionality."""
        
        def test_basic_functionality(self):
            """Test basic usage of YourClass."""
            obj = YourClass(param=42)
            result = obj.method()
            assert result == expected_value
        
        def test_edge_cases(self):
            """Test edge cases and error conditions."""
            with pytest.raises(ValueError):
                YourClass(invalid_param=-1)
        
        @pytest.mark.slow
        def test_performance_intensive(self):
            """Test that takes longer to run."""
            # Long-running test code

Using Our Test Runner
~~~~~~~~~~~~~~~~~~~

We provide a custom test runner with additional features:

.. code-block:: bash

    # Basic test run
    python tests/test_runner.py

    # Run with coverage reporting
    python tests/test_runner.py --coverage

    # Run only fast tests
    python tests/test_runner.py --mode fast

    # Check dependencies and run full suite
    python tests/test_runner.py --mode full

    # Check for missing dependencies
    python tests/test_runner.py --check-deps

Continuous Integration
--------------------

Our CI/CD pipeline runs automatically on every pull request and push to main.

CI Pipeline Overview
~~~~~~~~~~~~~~~~~

**Quick PR Check** (~2-3 minutes):
- Import validation
- Basic linting with ruff
- Fast unit tests
- Code formatting check

**Full CI Pipeline** (~10-15 minutes):
- Multi-version testing (Python 3.9-3.12)
- Cross-platform testing (Ubuntu, macOS, Windows)
- Complete test suite with coverage
- Security vulnerability scanning
- Performance benchmarks

**Workflow Files:**

- ``.github/workflows/pr-quick-check.yml``: Fast PR validation
- ``.github/workflows/ci.yml``: Comprehensive testing pipeline

Running CI Locally
~~~~~~~~~~~~~~~~~

Simulate CI checks on your local machine:

.. code-block:: bash

    # Run the same checks as CI
    python tests/test_runner.py --mode full

    # Check code quality (same as quick PR check)
    pre-commit run --all-files

    # Test across Python versions (if you have them)
    tox

    # Check import performance
    python -c "import time; start=time.time(); import dendritic_modeling; print(f'Import time: {time.time()-start:.2f}s')"

CI Results and Coverage
~~~~~~~~~~~~~~~~~~~~

CI automatically:

- **Posts coverage reports** as PR comments
- **Updates status badges** in README
- **Runs security scans** and reports vulnerabilities
- **Benchmarks performance** and detects regressions
- **Validates documentation** builds correctly

Code Quality Standards
--------------------

Code Style
~~~~~~~~

We enforce consistent code style using:

- **ruff**: Fast Python linter (replaces flake8, isort, etc.)
- **black**: Code formatter  
- **isort**: Import sorting
- **mypy**: Static type checking

.. code-block:: bash

    # Check and fix code style
    ruff check src/ tests/ --fix
    black src/ tests/
    isort src/ tests/

    # Type checking
    mypy src/

**Configuration Files:**

- ``ruff.toml``: Linting configuration
- ``pyproject.toml``: Tool configurations including pytest, coverage

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~

- All public APIs must have docstrings
- Use numpy-style docstrings
- Include examples in docstrings when helpful
- Update documentation when adding features

Testing Standards
~~~~~~~~~~~~~~~

- **Minimum 75% test coverage** required
- All new features must include tests
- Tests should be fast and deterministic
- Mock external dependencies in unit tests

Performance Standards
~~~~~~~~~~~~~~~~~~~

- No performance regressions in critical paths
- Memory usage should be reasonable
- Import time should be under 2 seconds

Contributing Guidelines
---------------------

Types of Contributions
~~~~~~~~~~~~~~~~~~~~

We welcome various types of contributions:

**Code Contributions:**
- New features and enhancements
- Bug fixes
- Performance improvements
- Code refactoring

**Documentation:**
- API documentation improvements
- Tutorial and example updates
- Installation guide improvements

**Testing:**
- Additional test cases
- Performance benchmarks
- Integration tests

**Infrastructure:**
- CI/CD improvements
- Build system enhancements
- Development tooling

**Research:**
- New dendritic algorithms
- Novel network architectures
- Performance comparisons

Contribution Areas
~~~~~~~~~~~~~~~~

**High Priority Areas:**

1. **Network Architectures**: New dendritic layer types
2. **Learning Algorithms**: Novel training strategies  
3. **Analysis Tools**: Better visualization and evaluation
4. **Performance**: Optimization and efficiency improvements
5. **Documentation**: Tutorials and examples

**Good First Issues:**

- Documentation improvements
- Adding test cases
- Fixing minor bugs
- Adding type hints
- Performance optimizations

Issue Reporting
~~~~~~~~~~~~~

When reporting bugs:

1. **Search existing issues** first
2. **Use issue templates** when available
3. **Provide minimal reproducible examples**
4. **Include system information** (OS, Python version, etc.)
5. **Describe expected vs actual behavior**

**Bug Report Template:**

.. code-block:: text

    **Bug Description**
    Clear description of the bug

    **To Reproduce**
    Steps to reproduce the behavior:
    1. Create model with config...
    2. Run training with...
    3. See error

    **Expected Behavior**
    What you expected to happen

    **Environment**
    - OS: [e.g. Ubuntu 20.04]
    - Python: [e.g. 3.11]
    - PyTorch: [e.g. 2.0.1]
    - Package version: [e.g. 0.1.0]

    **Additional Context**
    Stack trace, config files, etc.

Feature Requests
~~~~~~~~~~~~~~

For feature requests:

1. **Check if feature already exists**
2. **Describe the use case** clearly
3. **Provide examples** of desired API
4. **Consider implementation complexity**
5. **Discuss with maintainers** before starting large features

Review Process
~~~~~~~~~~~~

All contributions go through code review:

1. **Automated checks** must pass (CI, tests, linting)
2. **Maintainer review** for correctness and design
3. **Community feedback** welcome
4. **Documentation review** for user-facing changes
5. **Final approval** and merge

**Review Criteria:**

- Code quality and style
- Test coverage and quality
- Documentation completeness
- Performance impact
- API design consistency
- Backward compatibility

Getting Help
----------

**Communication Channels:**

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Code Reviews**: Feedback on pull requests

**Documentation:**

- **Installation Guide**: :doc:`../getting_started/installation`
- **Testing Guide**: ``tests/README.md``
- **CI Setup Guide**: ``CI_SETUP.md``
- **API Reference**: :doc:`../api/index`

**Development Resources:**

- **Test Runner**: ``python tests/test_runner.py --help``
- **CI Demo**: ``python demo_ci_setup.py``
- **Development Templates**: ``.github/`` directory

Recognition
---------

Contributors are recognized in:

- **Contributors file**: Automatic updates via all-contributors bot
- **Release notes**: Major contributions highlighted
- **GitHub insights**: Contribution statistics
- **Academic citations**: For research contributions

Thank you for contributing to the Dendritic Modeling Framework! Your contributions help advance the field of biologically-inspired neural networks.

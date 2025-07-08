Testing Guide
=============

The Dendritic Modeling Framework includes a comprehensive testing infrastructure to ensure code quality, reliability, and maintainability. This guide covers how to run tests, understand test organization, and write new tests.

Overview
--------

Our testing system includes:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark critical functionality
- **Code Quality Tests**: Linting, formatting, and type checking
- **Coverage Analysis**: Ensure adequate test coverage

**Test Statistics:**
- 1,500+ lines of test code across 7 test files
- 75% minimum coverage requirement
- Automated testing on every pull request
- Cross-platform and multi-version testing

Quick Test Commands
-----------------

.. code-block:: bash

    # Run all tests
    pytest

    # Run tests with coverage
    pytest --cov=dendritic_modeling

    # Run specific test categories
    pytest -m unit              # Unit tests only
    pytest -m integration       # Integration tests only
    pytest -m "not slow"        # Skip slow tests

    # Run our custom test runner
    python tests/test_runner.py

    # Run code quality checks
    pre-commit run --all-files

Test Organization
---------------

Test Structure
~~~~~~~~~~~~

Our tests are organized into logical categories:

.. code-block:: text

    tests/
    ├── test_analysis.py          # Analysis and evaluation tools (350+ lines)
    ├── test_datasets.py          # Dataset loading and processing (400+ lines)  
    ├── test_training_strategies.py # Training algorithms (150+ lines)
    ├── test_plotting.py          # Visualization functions (350+ lines)
    ├── test_config.py            # Configuration system (350+ lines)
    ├── test_runner.py            # Custom test runner (250+ lines)
    ├── README.md                 # Testing documentation
    └── pytest.ini               # Pytest configuration

Test Categories
~~~~~~~~~~~~~

Tests are marked with categories for selective execution:

**Unit Tests** (``@pytest.mark.unit``):
- Fast, isolated tests
- Mock external dependencies
- Test individual functions/classes

**Integration Tests** (``@pytest.mark.integration``):
- Test component interactions
- Use real dependencies where needed
- End-to-end workflow testing

**Slow Tests** (``@pytest.mark.slow``):
- Performance benchmarks
- Large dataset testing
- Time-intensive operations

**GPU Tests** (``@pytest.mark.requires_gpu``):
- Require CUDA-capable hardware
- Test GPU-specific functionality

**Data Tests** (``@pytest.mark.requires_data``):
- Require external datasets
- Network-dependent tests

Running Tests
-----------

Basic Test Execution
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Run all tests
    pytest

    # Verbose output with test names
    pytest -v

    # Run specific test file
    pytest tests/test_analysis.py

    # Run specific test function
    pytest tests/test_analysis.py::test_performance_analyzer_basic

    # Run tests matching pattern
    pytest -k "test_dataset"

Advanced Test Options
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Run tests in parallel (faster)
    pytest -n auto

    # Stop on first failure
    pytest -x

    # Show local variables on failure
    pytest -l

    # Run with timeout (prevent hanging)
    pytest --timeout=300

    # Only run failed tests from last run
    pytest --lf

    # Run tests that failed and were modified
    pytest --ff

Coverage Analysis
~~~~~~~~~~~~~~~

.. code-block:: bash

    # Run with coverage
    pytest --cov=dendritic_modeling

    # Generate HTML coverage report
    pytest --cov=dendritic_modeling --cov-report=html
    open htmlcov/index.html

    # Coverage with branch analysis
    pytest --cov=dendritic_modeling --cov-branch

    # Fail if coverage below threshold
    pytest --cov=dendritic_modeling --cov-fail-under=75

Custom Test Runner
~~~~~~~~~~~~~~~~

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

**Test Runner Features:**
- Dependency checking
- Performance monitoring
- Detailed error reporting
- Coverage analysis
- CI simulation

Test Configuration
----------------

Pytest Configuration
~~~~~~~~~~~~~~~~~~

Our ``pytest.ini`` file configures:

.. code-block:: ini

    [tool:pytest]
    testpaths = tests
    python_files = test_*.py *_test.py
    python_classes = Test*
    python_functions = test_*
    addopts = 
        --strict-markers
        --cov=dendritic_modeling
        --cov-report=term-missing
        --cov-fail-under=75
    markers =
        slow: marks tests as slow
        integration: marks tests as integration tests
        unit: marks tests as unit tests
        requires_gpu: marks tests that require GPU
        requires_data: marks tests that require external data

Coverage Configuration
~~~~~~~~~~~~~~~~~~~

Coverage settings in ``pyproject.toml``:

.. code-block:: toml

    [tool.coverage.run]
    source = ["src/dendritic_modeling"]
    omit = [
        "*/tests/*",
        "*/test_*.py",
        "*/__pycache__/*",
    ]

    [tool.coverage.report]
    exclude_lines = [
        "pragma: no cover",
        "def __repr__",
        "raise NotImplementedError",
        "if __name__ == .__main__.:",
    ]

Writing Tests
-----------

Test Structure
~~~~~~~~~~~~

Follow this structure when writing tests:

.. code-block:: python

    """Test module for YourComponent functionality."""

    import pytest
    from unittest.mock import Mock, patch
    from dendritic_modeling.your_module import YourClass


    class TestYourClass:
        """Test suite for YourClass."""
        
        def setup_method(self):
            """Set up test fixtures before each test method."""
            self.test_instance = YourClass(param=42)
        
        def test_basic_functionality(self):
            """Test basic usage of YourClass."""
            result = self.test_instance.method()
            assert result == expected_value
        
        def test_edge_cases(self):
            """Test edge cases and error conditions."""
            with pytest.raises(ValueError, match="Invalid parameter"):
                YourClass(invalid_param=-1)
        
        @pytest.mark.parametrize("input_val,expected", [
            (1, 2),
            (5, 10),
            (0, 0),
        ])
        def test_parameterized(self, input_val, expected):
            """Test multiple input/output combinations."""
            result = self.test_instance.calculate(input_val)
            assert result == expected
        
        @pytest.mark.slow
        def test_performance_intensive(self):
            """Test that takes longer to run."""
            # Long-running test code
            pass

Mocking Dependencies
~~~~~~~~~~~~~~~~~

Use mocking to isolate components:

.. code-block:: python

    from unittest.mock import Mock, patch, MagicMock

    class TestWithMocking:
        """Test class demonstrating mocking patterns."""
        
        @patch('dendritic_modeling.plotting.plt.show')
        def test_plotting_function(self, mock_show):
            """Test plotting without actually showing plots."""
            from dendritic_modeling.plotting import plot_results
            
            plot_results(data)
            mock_show.assert_called_once()
        
        def test_with_mock_model(self):
            """Test using a mock model."""
            mock_model = Mock()
            mock_model.forward.return_value = torch.tensor([1.0, 2.0])
            
            result = analyze_model(mock_model, test_data)
            mock_model.forward.assert_called_once()
            assert result is not None

Testing Neural Networks
~~~~~~~~~~~~~~~~~~~~

Special considerations for testing neural networks:

.. code-block:: python

    import torch
    from dendritic_modeling.networks import ExcitationInhibitionNetwork

    class TestEINetwork:
        """Test suite for EI networks."""
        
        def test_network_creation(self):
            """Test network can be created with valid parameters."""
            network = ExcitationInhibitionNetwork(
                input_dim=10,
                excitatory_layer_sizes=[20, 10],
                inhibitory_layer_sizes=[10, 5],
                # ... other parameters
            )
            assert network is not None
        
        def test_forward_pass_shape(self):
            """Test forward pass produces correct output shape."""
            network = ExcitationInhibitionNetwork(
                input_dim=10,
                output_dim=5,
                # ... other parameters
            )
            
            batch_size = 32
            input_tensor = torch.randn(batch_size, 10)
            output = network(input_tensor)
            
            assert output.shape == (batch_size, 5)
        
        def test_gradient_flow(self):
            """Test gradients flow correctly through network."""
            network = ExcitationInhibitionNetwork(
                input_dim=10,
                output_dim=5,
                # ... other parameters
            )
            
            input_tensor = torch.randn(1, 10, requires_grad=True)
            output = network(input_tensor)
            loss = output.sum()
            loss.backward()
            
            # Check gradients exist
            assert input_tensor.grad is not None
            for param in network.parameters():
                assert param.grad is not None

Testing Best Practices
---------------------

Test Design Principles
~~~~~~~~~~~~~~~~~~~~

1. **Fast and Reliable**: Tests should run quickly and consistently
2. **Independent**: Tests should not depend on each other
3. **Descriptive**: Test names should clearly describe what's being tested
4. **Focused**: Each test should test one specific behavior
5. **Maintainable**: Tests should be easy to understand and modify

Naming Conventions
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Good test names
    def test_network_forward_pass_with_valid_input():
        """Test network forward pass with valid input data."""
        pass

    def test_training_raises_error_with_invalid_lr():
        """Test training raises ValueError with negative learning rate."""
        pass

    def test_dataset_loader_returns_correct_batch_size():
        """Test dataset loader returns batches of correct size."""
        pass

Assertion Best Practices
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Use descriptive assertions
    assert result.shape == expected_shape, f"Expected {expected_shape}, got {result.shape}"

    # Test approximate equality for floats
    assert abs(result - expected) < 1e-6

    # Use pytest.approx for floating point comparisons
    assert result == pytest.approx(expected, rel=1e-5)

    # Test exception messages
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        train_model(lr=-0.1)

Test Data Management
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Use fixtures for reusable test data
    @pytest.fixture
    def sample_config():
        """Provide sample configuration for tests."""
        return {
            'model': {
                'network': {
                    'type': 'EINet',
                    'parameters': {
                        'input_dim': 784,
                        'output_dim': 10,
                    }
                }
            }
        }

    # Use parametrize for multiple test cases
    @pytest.mark.parametrize("batch_size,expected_batches", [
        (32, 32),
        (64, 16),
        (128, 8),
    ])
    def test_dataloader_batch_sizes(self, batch_size, expected_batches):
        """Test dataloader with different batch sizes."""
        # Test implementation

Continuous Integration
--------------------

Local CI Simulation
~~~~~~~~~~~~~~~~~

Run the same checks as CI locally:

.. code-block:: bash

    # Full CI simulation
    python tests/test_runner.py --mode full

    # Quick checks (same as PR quick check)
    pre-commit run --all-files

    # Test import performance
    python -c "import time; start=time.time(); import dendritic_modeling; print(f'Import: {time.time()-start:.2f}s')"

CI Test Matrix
~~~~~~~~~~~~

Our CI runs tests across:

- **Python versions**: 3.9, 3.10, 3.11, 3.12
- **Operating systems**: Ubuntu, macOS, Windows
- **PyTorch versions**: Latest stable versions
- **Dependency combinations**: Core, full, minimal

Performance Testing
~~~~~~~~~~~~~~~~~

We monitor:

- **Test execution time**: Prevent slow test regressions
- **Import time**: Keep imports under 2 seconds
- **Memory usage**: Monitor memory consumption
- **Coverage**: Maintain minimum 75% coverage

Troubleshooting Tests
-------------------

Common Issues
~~~~~~~~~~~

**Import Errors**

.. code-block:: bash

    # Check package installation
    pip show dendritic_modeling

    # Reinstall in development mode
    pip install -e .

**Slow Tests**

.. code-block:: bash

    # Skip slow tests during development
    pytest -m "not slow"

    # Profile test execution
    pytest --durations=10

**GPU Test Failures**

.. code-block:: bash

    # Skip GPU tests if no GPU available
    pytest -m "not requires_gpu"

    # Check CUDA availability
    python -c "import torch; print(torch.cuda.is_available())"

**Coverage Issues**

.. code-block:: bash

    # Check which lines are missing coverage
    pytest --cov=dendritic_modeling --cov-report=html
    open htmlcov/index.html

    # Run with branch coverage
    pytest --cov=dendritic_modeling --cov-branch

Debug Test Failures
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Run with verbose output
    pytest -v -s

    # Drop into debugger on failure
    pytest --pdb

    # Show local variables on failure
    pytest -l

    # Only run failed tests
    pytest --lf

Getting Help
----------

**Resources:**

- **Test Documentation**: ``tests/README.md``
- **CI Setup Guide**: ``CI_SETUP.md``
- **Contributing Guide**: :doc:`contributing`
- **GitHub Issues**: Report test-related bugs

**Common Commands Reference:**

.. code-block:: bash

    # Essential test commands
    pytest                              # Run all tests
    pytest -v                          # Verbose output
    pytest -k "test_name"              # Run specific tests
    pytest -m unit                     # Run unit tests only
    pytest --cov=dendritic_modeling    # Run with coverage
    python tests/test_runner.py        # Custom test runner
    pre-commit run --all-files         # Code quality checks

The testing infrastructure ensures the reliability and maintainability of the Dendritic Modeling Framework. By following these guidelines, you can contribute high-quality tests that help maintain the project's quality standards. 
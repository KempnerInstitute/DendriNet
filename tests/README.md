# Test Suite

Comprehensive test suite for the dendritic modeling framework.

## Test Structure

```
tests/
├── README.md                    # This file
├── test_runner.py              # Custom test runner with advanced features
├── run_dendritic_conv2d_tests.py # Test runner for dendritic convolution tests
├── test_analysis.py            # Tests for analysis modules
├── test_datasets.py            # Tests for dataset loading and processing
├── test_training_strategies.py # Tests for training strategies and optimizers
├── test_plotting.py            # Tests for visualization functions
├── test_config.py              # Tests for configuration handling
├── test_models.py              # Tests for model classes (existing)
├── test_networks.py            # Tests for network architectures (existing)
├── test_utils.py               # Tests for utility functions (existing)
├── test_dendritic_conv2d.py    # Tests for dendritic convolution layers
├── test_*.py                   # Additional test modules
└── __init__.py                 # Test package initialization
```

## Quick Start

### Basic Test Running

```bash
# Run all tests
python -m pytest

# Run all tests with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_models.py

# Run specific test class
python -m pytest tests/test_models.py::TestProbabilisticClassifier

# Run specific test method
python -m pytest tests/test_models.py::TestProbabilisticClassifier::test_predict_deterministic
```

### Using the Custom Test Runner

```bash
# Run all tests
python tests/test_runner.py

# Run only unit tests
python tests/test_runner.py --test-type unit

# Run with coverage report
python tests/test_runner.py --test-type coverage

# Run specific modules
python tests/test_runner.py --modules models networks

# Run with HTML coverage report
python tests/test_runner.py --test-type coverage --html-coverage

# Check testing dependencies
python tests/test_runner.py --check-deps

# Run dendritic convolution test suite
python tests/run_dendritic_conv2d_tests.py
```

## Coverage Reports

### Generate Coverage Reports

```bash
# Terminal coverage report
python -m pytest --cov=src/dendritic_modeling --cov-report=term-missing

# HTML coverage report
python -m pytest --cov=src/dendritic_modeling --cov-report=html

# XML coverage report (for CI/CD)
python -m pytest --cov=src/dendritic_modeling --cov-report=xml
```

The HTML report will be generated in `htmlcov/index.html` and can be opened in a browser.

## Test Categories

### Unit Tests
Test individual components in isolation:
- Models: `test_models.py` - Test model classes and their methods
- Networks: `test_networks.py` - Test network architectures and layers
- Datasets: `test_datasets.py` - Test dataset loading and processing
- Analysis: `test_analysis.py` - Test analysis modules and functions
- Training: `test_training_strategies.py` - Test training strategies and optimizers
- Plotting: `test_plotting.py` - Test visualization functions
- Config: `test_config.py` - Test configuration handling
- Utils: `test_utils.py` - Test utility functions

### Integration Tests
Test interactions between components:
- End-to-End: Complete training workflows
- Cross-Module: Interactions between different modules

### Performance Tests
Test performance characteristics:
- Benchmarks: Performance benchmarks for critical paths
- Memory: Memory usage tests
- GPU: GPU-specific functionality (when available)

## Test Markers

Tests are marked with pytest markers for selective running:

```bash
# Run only unit tests
python -m pytest -m unit

# Run only integration tests
python -m pytest -m integration

# Skip slow tests
python -m pytest -m "not slow"

# Run only tests that don't require GPU
python -m pytest -m "not requires_gpu"

# Run only tests that don't require external data
python -m pytest -m "not requires_data"
```

Available markers:
- `unit`: Unit tests
- `integration`: Integration tests
- `slow`: Slow-running tests
- `requires_gpu`: Tests requiring GPU
- `requires_data`: Tests requiring external data

## Development Workflow

### Writing New Tests

1. Choose the appropriate test file based on the module you're testing
2. Follow naming conventions:
   - Test files: `test_<module_name>.py`
   - Test classes: `Test<ClassName>`
   - Test methods: `test_<functionality>`

3. Use proper test structure:
```python
class TestYourComponent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_basic_functionality(self):
        """Test basic functionality with clear docstring."""
        # Arrange
        # Act
        # Assert
        pass
```

4. Add appropriate markers:
```python
import pytest

@pytest.mark.slow
def test_long_running_operation(self):
    """Test that takes a long time to run."""
    pass

@pytest.mark.requires_gpu
def test_gpu_functionality(self):
    """Test that requires GPU."""
    pass
```

### Running Tests During Development

```bash
# Run tests for the module you're working on
python -m pytest tests/test_models.py -v

# Run tests and stop on first failure
python -m pytest tests/test_models.py -x

# Run tests with coverage for specific module
python -m pytest tests/test_models.py --cov=src/dendritic_modeling.models

# Run tests in parallel (if pytest-xdist is installed)
python -m pytest -n auto
```

### Pre-commit Testing

Before committing code, run the full test suite:

```bash
# Quick test run (skip slow tests)
python -m pytest -m "not slow"

# Full test run with coverage
python tests/test_runner.py --test-type coverage

# Check test dependencies
python tests/test_runner.py --check-deps
```

## Configuration

### Pytest Configuration

The project uses `pytest.ini` for configuration:
- Minimum coverage threshold: 75%
- Automatic test discovery
- Warning filters
- Marker definitions

### Environment Variables

Set these environment variables for testing:

```bash
# Disable CUDA for consistent CPU testing
export CUDA_VISIBLE_DEVICES=""

# Set random seeds for reproducible tests
export PYTHONHASHSEED=42

# Reduce matplotlib warnings
export MPLBACKEND=Agg
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e .[dev]
    - name: Run tests
      run: |
        python tests/test_runner.py --test-type coverage
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Debugging Tests

### Running Individual Tests

```bash
# Run with debugger
python -m pytest tests/test_models.py::test_specific_function --pdb

# Run with verbose output and no capture
python -m pytest tests/test_models.py -v -s

# Run with detailed output on failures
python -m pytest tests/test_models.py --tb=long
```

### Common Issues and Solutions

1. Import Errors:
   - Ensure you're running tests from the project root
   - Check that the package is properly installed: `pip install -e .`

2. Missing Dependencies:
   - Install test dependencies: `pip install pytest pytest-cov`
   - Check dependencies: `python tests/test_runner.py --check-deps`

3. Slow Tests:
   - Skip slow tests: `python -m pytest -m "not slow"`
   - Run specific modules: `python -m pytest tests/test_utils.py`

4. GPU Tests Failing:
   - Skip GPU tests: `python -m pytest -m "not requires_gpu"`
   - Set CPU-only mode: `export CUDA_VISIBLE_DEVICES=""`

## Test Checklist

Before submitting code, ensure:

- [ ] All existing tests pass
- [ ] New functionality has corresponding tests
- [ ] Test coverage meets minimum threshold (75%)
- [ ] Tests are properly documented
- [ ] Tests use appropriate markers
- [ ] No test warnings or errors
- [ ] Tests run in reasonable time

## Contributing

When contributing tests:

1. Follow existing patterns in the test files
2. Add comprehensive docstrings explaining what each test does
3. Use descriptive test names that clearly indicate what is being tested
4. Mock external dependencies when appropriate
5. Test both success and failure cases
6. Add markers for test categorization
7. Keep tests independent - no test should depend on another test's state

## Additional Resources

- [PyTest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [Testing Best Practices](https://realpython.com/python-testing/)

---

For questions about the test suite, please refer to the documentation or create an issue in the repository.

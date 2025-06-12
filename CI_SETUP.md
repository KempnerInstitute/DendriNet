# Continuous Integration (CI) Setup Guide

This document explains how to set up and use the automated testing system for the dendritic modeling project.

## 🚀 Quick Start

Once you push this configuration to your GitHub repository, **testing will automatically run on every Pull Request**. No additional setup required!

## 📋 What Gets Tested Automatically

### On Every PR:

1. **🚀 Quick PR Check** (~2-3 minutes)
   - Fast unit tests only
   - Basic import checks
   - Code linting (ruff, black, isort)
   - Provides immediate feedback

2. **🧪 Full CI Pipeline** (~10-15 minutes) 
   - Complete test suite across multiple Python versions (3.9-3.12)
   - Cross-platform testing (Ubuntu, macOS, Windows)
   - Coverage report generation
   - Security scanning
   - Documentation building

3. **📊 Automatic Reports**
   - Coverage reports posted as PR comments
   - Test result artifacts
   - Linting feedback directly in PR

## 🛠️ CI Workflows Explained

### 1. Quick PR Check (`.github/workflows/pr-quick-check.yml`)

**Purpose**: Provide rapid feedback on PRs

**What it does**:
```bash
# Runs fast tests only
pytest -m "not slow and not requires_gpu and not requires_data"

# Basic linting
ruff check src/ tests/
black --check src/ tests/
isort --check src/ tests/

# Import validation  
python -c "import dendritic_modeling"
```

**When it runs**: Immediately when PR is opened/updated

### 2. Full CI Pipeline (`.github/workflows/ci.yml`)

**Purpose**: Comprehensive testing and validation

**Test Matrix**:
- **Python versions**: 3.9, 3.10, 3.11, 3.12
- **Operating systems**: Ubuntu (all versions), macOS (3.11), Windows (3.11)
- **Test types**: Unit, integration, performance, security

**Features**:
- ✅ Full test suite with coverage
- ✅ Cross-platform compatibility 
- ✅ Security vulnerability scanning
- ✅ Performance regression detection
- ✅ Documentation building
- ✅ Artifact uploads

## 🔧 Setting Up for Your Repository

### 1. Repository Setup

```bash
# 1. Ensure all CI files are in your repository
git add .github/workflows/
git add pytest.ini ruff.toml .pre-commit-config.yaml
git commit -m "Add CI configuration"
git push
```

### 2. GitHub Repository Settings

#### Required Actions:
1. **Enable GitHub Actions** (usually enabled by default)
2. **Add branch protection rules** (recommended):
   ```
   Repository Settings → Branches → Add rule
   - Branch name pattern: main (or master)
   - Require status checks: ✅
   - Require branches to be up to date: ✅
   - Required status checks: 
     * Quick PR Check
     * test (ubuntu-latest, 3.11)
   ```

#### Optional Integrations:

**Codecov Setup** (for enhanced coverage reporting):
```bash
# 1. Go to https://codecov.io
# 2. Sign up with GitHub
# 3. Add your repository
# 4. No additional secrets needed - uses GITHUB_TOKEN
```

**Dependabot Setup** (for automated dependency updates):
```yaml
# Create .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
```

### 3. Local Development Setup

#### Install Pre-commit Hooks:
```bash
# Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install

# Run hooks on all files (optional)
pre-commit run --all-files
```

#### Local Testing Commands:
```bash
# Run the same tests as CI
python tests/test_runner.py --test-type coverage

# Quick tests (like PR check)
python -m pytest -m "not slow and not requires_gpu" -q

# Full test suite
python -m pytest tests/ --cov=src/dendritic_modeling

# Linting (same as CI)
ruff check src/ tests/
black src/ tests/
isort src/ tests/
```

## 📊 Understanding CI Results

### 1. PR Status Checks

After opening a PR, you'll see status checks:

```
✅ Quick PR Check  
⏳ CI Tests / test (ubuntu-latest, 3.11)
⏳ CI Tests / test (ubuntu-latest, 3.12)  
⏳ CI Tests / lint-and-format
⏳ CI Tests / security-scan
```

### 2. Coverage Reports

Coverage reports are automatically posted as PR comments:

```markdown
## 📊 Coverage Report
Current coverage: 85.2%
- src/dendritic_modeling/models/: 92%
- src/dendritic_modeling/networks/: 78%
- src/dendritic_modeling/analysis/: 81%
```

### 3. Artifact Downloads

Test results and coverage reports are available as downloadable artifacts:
- `test-results-ubuntu-latest-3.11.zip`
- `coverage.xml`
- `htmlcov/` (browsable HTML coverage report)

## 🚨 Troubleshooting Common Issues

### Test Failures

**Import Errors**:
```bash
# Check package installation
python -c "import dendritic_modeling"

# Verify PYTHONPATH
echo $PYTHONPATH
```

**GPU Tests Failing**:
```bash
# Tests should skip GPU by default in CI
# If needed, mark tests with @pytest.mark.requires_gpu
```

**Slow Tests Timing Out**:
```bash
# Mark long tests with @pytest.mark.slow
# They'll be skipped in quick checks
```

### Linting Failures

**Fix automatically**:
```bash
# Auto-fix most issues
ruff check src/ tests/ --fix
black src/ tests/
isort src/ tests/
```

**Common issues**:
- Line too long → Black handles this
- Import sorting → Use `isort`
- Unused variables → Use `_` prefix or remove

### Coverage Issues

**Increase coverage**:
```bash
# See what's not covered
python -m pytest --cov=src/dendritic_modeling --cov-report=html
# Open htmlcov/index.html in browser
```

**Exclude files from coverage** (in pytest.ini):
```ini
[tool:pytest]
addopts = --cov-omit="*/tests/*,*/conftest.py"
```

## 🎯 Best Practices

### For Contributors

1. **Before Creating PR**:
   ```bash
   # Run quick local tests
   python tests/test_runner.py --modules your_module
   
   # Check linting
   ruff check src/ tests/
   black --check src/ tests/
   ```

2. **Writing Tests**:
   ```python
   import pytest
   
   @pytest.mark.slow
   def test_expensive_operation():
       """Mark slow tests to exclude from quick checks."""
       pass
   
   @pytest.mark.requires_gpu  
   def test_gpu_functionality():
       """Mark GPU tests for conditional running."""
       pass
   ```

3. **PR Description**:
   - Use the provided PR template
   - Link to related issues
   - Describe testing performed

### For Maintainers

1. **Branch Protection**:
   - Require status checks before merge
   - Require up-to-date branches
   - Consider requiring code review

2. **Release Process**:
   ```bash
   # Tag releases to trigger additional workflows
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```

3. **Monitoring**:
   - Review failed CI runs regularly
   - Monitor coverage trends
   - Update dependencies via Dependabot

## 📈 Advanced Configuration

### Custom Test Markers

Add to `pytest.ini`:
```ini
markers =
    integration: marks tests as integration tests
    gpu: marks tests as requiring GPU
    slow: marks tests as slow (>10s)
    requires_data: marks tests requiring external data
```

### Environment Variables

Set in GitHub repository secrets for sensitive data:
```yaml
# In workflow files
env:
  CUSTOM_API_KEY: ${{ secrets.CUSTOM_API_KEY }}
```

### Performance Benchmarking

Add `pytest-benchmark` for performance regression detection:
```bash
pip install pytest-benchmark
pytest tests/ --benchmark-only
```

## 🔗 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Ruff Linter Documentation](https://beta.ruff.rs/docs/)
- [Pre-commit Documentation](https://pre-commit.com/)

---

## 📞 Support

If you encounter issues with the CI setup:

1. Check the **Actions** tab in your GitHub repository
2. Review the **test logs** for specific error messages
3. Run tests locally using the same commands as CI
4. Consult this documentation for troubleshooting steps

**Happy testing! 🧪✨** 
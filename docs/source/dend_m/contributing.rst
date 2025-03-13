Contributing to Dendritic Modeling
================================

This guide explains how to contribute to the dendritic modeling codebase, whether you're reporting bugs, suggesting improvements, or submitting code contributions.

Opening Issues
-------------

Issues are a great way to report bugs, request features, or ask questions about the codebase.

When opening an issue:

1. **Check existing issues** first to avoid duplicates
2. **Use descriptive titles** that clearly indicate the problem or request
3. **Provide complete information**, including:
   - Steps to reproduce bugs
   - Expected vs. actual behavior
   - Error messages (if applicable)
   - Environment details (OS, Python version, etc.)
   - Screenshots or code examples when relevant

4. **Use labels** to categorize your issue (e.g., bug, enhancement, documentation)

Example template for bug reports:

.. code-block:: text

    ## Bug Description
    [Clear description of the issue]

    ## Steps to Reproduce
    1. [First step]
    2. [Second step]
    3. [And so on...]

    ## Expected Behavior
    [What you expected to happen]

    ## Actual Behavior
    [What actually happened]

    ## Environment
    - OS: [e.g., Ubuntu 22.04]
    - Python version: [e.g., 3.10.5]
    - Dependencies: [Any relevant package versions]

Submitting Pull Requests
-----------------------

Pull requests (PRs) allow you to contribute code changes to the project.

### Before Submitting a PR

1. **Discuss major changes** by opening an issue first
2. **Fork the repository** to your personal account
3. **Create a feature branch** with a descriptive name
4. **Follow coding standards** described in the Kempner Computing Handbook
5. **Write or update tests** for your changes
6. **Ensure tests pass** before submitting

### PR Submission Guidelines

1. **Keep PRs focused** on a single issue or feature
2. **Write a clear PR description** explaining your changes
3. **Reference related issues** using the `#issue-number` syntax
4. **Update documentation** to reflect your changes
5. **Include screenshots** for UI-related changes

Example PR template:

.. code-block:: text

    ## Description
    [Describe what your PR does]

    ## Related Issues
    Fixes #[issue number]

    ## Type of Change
    - [ ] Bug fix
    - [ ] New feature
    - [ ] Documentation update
    - [ ] Performance improvement
    - [ ] Code refactoring

    ## Testing
    [Describe the tests you ran]

    ## Screenshots (if applicable)
    [Add screenshots here]

    ## Checklist
    - [ ] My code follows the project style guidelines
    - [ ] I have performed a self-review of my code
    - [ ] I have commented my code in hard-to-understand areas
    - [ ] I have updated the documentation
    - [ ] My changes generate no new warnings
    - [ ] I have added tests that prove my fix is effective or that my feature works
    - [ ] New and existing tests pass with my changes

Following Kempner Computing Handbook Guidelines
----------------------------------------------

The Kempner Computing Handbook provides important guidelines that should be followed when contributing to this project:

1. **Code Organization**
   - Keep code modular and well-organized
   - Follow the established project structure 
   - Maintain clear separation of concerns

2. **Documentation Standards**
   - Document all public functions, classes, and modules
   - Include examples in docstrings
   - Keep documentation up-to-date with code changes

3. **Testing Requirements**
   - Write unit tests for new functionality
   - Ensure test coverage meets project standards
   - Include integration tests for complex components

4. **Code Quality**
   - Follow PEP 8 style guidelines
   - Use type hints where appropriate
   - Keep functions and methods focused on a single responsibility
   - Avoid redundant or duplicate code

5. **Version Control Practices**
   - Write clear, concise commit messages
   - Make frequent, small commits
   - Rebase feature branches before submission

6. **Review Process**
   - Be responsive to reviewer feedback
   - Address all review comments before merge
   - Be constructive and respectful in code reviews

Development Setup
----------------

To set up your development environment:

1. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/yourusername/dendritic-modeling.git
      cd dendritic-modeling

2. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install development dependencies**:

   .. code-block:: bash

      pip install -e ".[dev]"

4. **Install pre-commit hooks**:

   .. code-block:: bash

      pre-commit install

5. **Run tests to ensure everything is working**:

   .. code-block:: bash

      pytest

Getting Help
-----------

If you're unsure about anything related to contributing:

1. **Read the documentation** thoroughly
2. **Check existing issues and discussions**
3. **Ask questions** in the project's communication channels
4. **Reach out to maintainers** for clarification on complex issues

Thank you for contributing to the dendritic modeling framework! 
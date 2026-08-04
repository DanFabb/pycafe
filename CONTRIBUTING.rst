Contributing Guide
==================

Contributions to **pyCAFE** are welcome and greatly appreciated.
Bug fixes, improvements, and new features are all encouraged.

This document outlines the recommended workflow for contributing to the
development of pyCAFE.

---

Workflow
--------

Contributions are submitted via pull requests.
Each pull request should ideally address **one bug fix or one enhancement**.
Keeping changes focused makes them easier to review and maintain.

The typical development workflow is as follows.

1. Fork the repository
~~~~~~~~~~~~~~~~~~~~~~

Fork the ``pycafe`` repository to your own GitHub account.

2. Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~

Clone your fork to your local development machine:

.. code-block:: console

    git clone https://github.com/DanFabb/pycafe.git
    cd pycafe

3. Create a development branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new branch for your bug fix or feature:

.. code-block:: console

    git checkout -b name-of-your-bugfix-or-feature

All development should take place on this branch.

---

Development
-----------

4. Implement your changes
~~~~~~~~~~~~~~~~~~~~~~~~~

- Modify or extend the relevant modules, functions, or classes.
- Follow the existing code style and structure.
- Keep changes as small and focused as possible.

5. Add or update tests
~~~~~~~~~~~~~~~~~~~~~~

All non-trivial changes should be accompanied by appropriate unit tests.

- Add new tests or update existing ones in the ``tests/`` directory.
- Ensure that the test suite passes:

.. code-block:: console

    pytest

---

Documentation
-------------

6. Update documentation
~~~~~~~~~~~~~~~~~~~~~~~

If your change affects the user-facing API, workflow, or behavior,
the documentation should be updated accordingly.

To build the documentation locally:

.. code-block:: console

    cd docs
    make clean
    make html

Verify that the documentation builds without errors and that the rendered
HTML correctly reflects your changes.

---

Submitting the contribution
----------------------------

7. Commit and push your changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Commit your changes with a clear and descriptive commit message:

.. code-block:: console

    git add .
    git commit -m "Brief description of the change"
    git push origin name-of-your-bugfix-or-feature

Whenever possible, keep the pull request history clean by squashing commits
into a single logical commit.

8. Open a pull request
~~~~~~~~~~~~~~~~~~~~~~

Submit a pull request against the ``main`` branch of the pyCAFE repository
using the hosting service (e.g. GitHub).

Please ensure that:

- Automated tests pass
- Documentation builds successfully
- The pull request description clearly explains the motivation and scope
  of the change

---

Code style and design philosophy
--------------------------------

pyCAFE aims to be:

- Clear and readable
- Modular and extensible
- Explicit in its numerical workflow

Contributors are encouraged to preserve these principles when adding new
features or refactoring existing code.

---

License
-------

By contributing to pyCAFE, you agree that your contributions will be licensed
under the MIT License.
See the ``LICENSE`` file for details.

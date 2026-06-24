# Contributing

Contributions to openKARST are welcome, including bug reports, documentation
improvements, feature proposals, and code changes.

## Report a problem

Use the [GitHub issue tracker](https://github.com/ERC-Karst/openkarst/issues) to
report reproducible bugs or propose new functionality. Before opening an issue,
check whether a related report or discussion already exists.

For bug reports, include:

- the openKARST and Python versions;
- a minimal example that reproduces the problem;
- the expected and observed behavior;
- the complete error message, when applicable.

## Submit a change

Code and documentation changes can be proposed through a
[pull request](https://github.com/ERC-Karst/openkarst/pulls). Keep each pull
request focused on one change and explain its purpose and user-visible effect.

When changing model behavior, add or update the relevant tests. Run the test
suite before submitting:

```bash
pytest
```

## Improve the documentation

The manual is built with MkDocs. From the repository root, preview documentation
changes with:

```bash
pyenv exec mkdocs serve
```

Check the production build before submitting:

```bash
pyenv exec mkdocs build --strict
```

Documentation should use sentence-case headings, concise examples, and
terminology consistent with the rest of the manual.

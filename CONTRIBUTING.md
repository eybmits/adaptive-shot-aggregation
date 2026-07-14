# Contributing

This repository is a frozen research artifact. Corrections that improve
reproducibility, documentation, portability, or test coverage are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Run the tests before opening a pull request:

```bash
make test
make verify
```

Changes to figures, manuscript values, or frozen result files must also pass:

```bash
make all
```

## Research artifact rules

1. Do not replace frozen evidence without updating its manifest and checksums.
2. Do not treat repeated shots or grouped views as independent forecasting
   units.
3. Keep the chronological validation and test boundary intact.
4. Document changes that alter a reported number or figure.
5. Never commit credentials, local paths, caches, or raw service tokens.

Bug reports should include the command, Python version, operating system, and
the complete error message needed to reproduce the problem.

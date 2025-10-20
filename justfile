#!/usr/bin/env just --justfile

# A justfile is a place for project-specific commands that you have copy/pasted in a note somewhere.
# Install this (on macOS) with `brew install just` and then run `just --list` in this repo.

# List commands
default:
  just --list

# Says hello!
hello:
  echo "Hello World from" $(pwd)

# Run tests on a subproject (default is qcore)

test:
  uv run pytest --cov=python-template tests --cov-config=.coveragerc

# Run type cheking on a subproject
type:
  uv run --frozen mypy src/ --non-interactive

# Lint codebase
lint:
    uv run pylint check src

# Format codebase
format:
    uv run black src

# Check the forecast
weather:
  #!/usr/bin/env uv run --script
  # /// script
  # dependencies = [ "requests" ]
  # ///
  import requests
  data = requests.get("https://wttr.in/?F")
  print(data.text)

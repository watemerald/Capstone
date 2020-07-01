#!/bin/sh
set -e

python -m flake8 .
python -m isort ./*/*.py --check-only
python -m black . --check
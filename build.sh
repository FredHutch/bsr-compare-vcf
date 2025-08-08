#!/bin/bash
set -e

rm -rf test_build

uv run marimo export html-wasm app.py -o test_build --mode run --no-show-code

uv run python -m http.server --directory test_build

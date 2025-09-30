#!/usr/bin/env bash

set -eu

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
cd "$ROOT_DIR"

git ls-files '*.cpp' '*.hpp' | xargs -L 1 clang-format -i

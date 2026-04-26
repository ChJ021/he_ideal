#!/usr/bin/env bash
set -euo pipefail

OPENFHE_ROOT="${OPENFHE_ROOT:-/home/cj/github/openfhe}"
OPENFHE_INSTALL_DIR="${OPENFHE_INSTALL_DIR:-${OPENFHE_ROOT}/install}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${HETUNE_OPENFHE_RUNNER_BUILD_DIR:-${REPO_ROOT}/build/openfhe_runner}"
SOURCE_DIR="${REPO_ROOT}/native/openfhe_runner"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${OPENFHE_INSTALL_DIR}"

cmake --build "${BUILD_DIR}" --parallel

echo "Built ${BUILD_DIR}/hetune_openfhe_runner"

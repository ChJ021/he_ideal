#!/usr/bin/env bash
set -euo pipefail

OPENFHE_ROOT="${OPENFHE_ROOT:-/home/cj/github/openfhe}"
OPENFHE_REPO="${OPENFHE_REPO:-https://github.com/openfheorg/openfhe-development.git}"
OPENFHE_BRANCH="${OPENFHE_BRANCH:-main}"

SRC_DIR="${OPENFHE_ROOT}/src"
BUILD_DIR="${OPENFHE_ROOT}/build"
INSTALL_DIR="${OPENFHE_ROOT}/install"

mkdir -p "${OPENFHE_ROOT}"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
  git clone --recursive --branch "${OPENFHE_BRANCH}" "${OPENFHE_REPO}" "${SRC_DIR}"
else
  git -C "${SRC_DIR}" fetch --all --tags
  git -C "${SRC_DIR}" submodule update --init --recursive
fi

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_UNITTESTS=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DWITH_OPENMP=ON

cmake --build "${BUILD_DIR}" --parallel
cmake --install "${BUILD_DIR}"

echo "OpenFHE installed under ${INSTALL_DIR}"

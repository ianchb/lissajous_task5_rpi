#!/usr/bin/env bash
set -euo pipefail

build_dir="${BUILD_DIR:-build}"
jobs="${JOBS:-4}"

cmake -S . -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" -j"${jobs}"
ctest --test-dir "${build_dir}" --output-on-failure

sudo cmake --install "${build_dir}" --prefix /usr/local
sudo install -m 0644 docs/task5-autostart.service \
    /etc/systemd/system/task5-autostart.service
sudo systemctl daemon-reload
sudo systemctl enable --now task5-autostart.service

systemctl --no-pager --full status task5-autostart.service

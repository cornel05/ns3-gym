#!/usr/bin/env bash
set -euo pipefail

# Conservative setup script for ns-3.40 + ns3-gym (app-ns-3.36+).
# This is a documented helper, not an unattended installer.

NS3_VER="3.40"
NS3_ALLINONE="ns-allinone-${NS3_VER}"
NS3_TARBALL="${NS3_ALLINONE}.tar.bz2"
NS3_URL="https://www.nsnam.org/releases/${NS3_TARBALL}"

echo "[1/6] Install basic Linux packages"
echo "Run the following command yourself (sudo may prompt for password):"
echo "  sudo apt update && sudo apt install -y build-essential cmake git python3 python3-pip python3-venv pkg-config"

echo

echo "[2/6] Download and extract ns-allinone-${NS3_VER}"
if [[ ! -f "${NS3_TARBALL}" ]]; then
  wget "${NS3_URL}"
fi
tar -xjf "${NS3_TARBALL}"

cd "${NS3_ALLINONE}/ns-${NS3_VER}"

echo "[3/6] Clone ns3-gym into contrib/opengym"
mkdir -p contrib
if [[ ! -d contrib/opengym/.git ]]; then
  git clone https://github.com/tkn-tub/ns3-gym.git contrib/opengym
fi

cd contrib/opengym

echo "[4/6] Checkout target branch app-ns-3.36+"
git fetch origin
git checkout app-ns-3.36+

cd ../../

echo "[5/6] Configure and build ns-3"
./ns3 configure
./ns3 build

echo "[6/6] Install ns3gym Python package"
cd contrib/opengym/model/ns3gym

if [[ ! -f ns3gym/messages_pb2.py ]]; then
  echo "Missing generated protobuf file: contrib/opengym/model/ns3gym/ns3gym/messages_pb2.py"
  echo "Run ./ns3 configure && ./ns3 build first, then retry this install step."
  exit 1
fi

python -m pip install -e .

echo "Done. Verify ns3gym import in your Python environment before training."

#!/bin/bash
set -e

echo "--- Setting up environment for ROCm PyTorch ---"

echo "--- Setting up environment for manual libtorch ---"
export LIBTORCH="$(pwd)/libtorch"
export LD_LIBRARY_PATH="/opt/rocm/lib:$LIBTORCH/lib:$LD_LIBRARY_PATH"
echo "LIBTORCH=$LIBTORCH"

# Disable SDMA to prevent driver crashes/hangs on consumer RDNA2 cards
export HSA_ENABLE_SDMA=0

cargo run --release -- "$@"
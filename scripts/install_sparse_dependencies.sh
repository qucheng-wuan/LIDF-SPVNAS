#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dependency_dir="$project_dir/.deps"
mkdir -p "$dependency_dir"
: "${CUDA_HOME:?Set CUDA_HOME to a CUDA 12.1 toolkit containing bin/nvcc}"
export PATH="$CUDA_HOME/bin:$PATH"
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-2}"
# Override this for GPUs other than the RTX 3090 used in the experiments.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

if [ ! -d "$dependency_dir/sparsehash/.git" ]; then
    git clone https://github.com/sparsehash/sparsehash.git "$dependency_dir/sparsehash"
fi
git -C "$dependency_dir/sparsehash" checkout --detach f93c0c69e959c1c77611d0ba8d107aa971338811
(
    cd "$dependency_dir/sparsehash"
    ./configure --prefix="$dependency_dir/sparsehash_install"
    make -j "$MAX_JOBS"
    make install
)
export CPLUS_INCLUDE_PATH="$dependency_dir/sparsehash_install/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
if [ ! -d "$dependency_dir/torchsparse/.git" ]; then
    git clone https://github.com/mit-han-lab/torchsparse.git "$dependency_dir/torchsparse"
fi
git -C "$dependency_dir/torchsparse" checkout --detach 385f5ce8718fcae93540511b7f5832f4e71fd835
python -m pip install --no-deps --no-build-isolation "$dependency_dir/torchsparse"
python "$project_dir/scripts/spvnas/check_sparse_api.py"

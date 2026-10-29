#!/usr/bin/env bash
set -euo pipefail
cd /home/quchengzou/implicit_depth
export PATH=/home/quchengzou/miniconda3/envs/lidf/bin:$PATH
export CUDA_HOME=/home/quchengzou/miniconda3/envs/lidf
export TORCH_CUDA_ARCH_LIST=8.6
export FORCE_CUDA=1
export MAX_JOBS=2
export CPLUS_INCLUDE_PATH=/home/quchengzou/implicit_depth/third_party/sparsehash_install/include
# Pin the API actually used by the July 2024 SPVNAS source. The v2.1.0 tag
# e2a6c32 is older and lacks GPUHashTable/TensorCache; it is NOT used here.
test "$(git -C third_party/torchsparse_current rev-parse HEAD)" = 385f5ce8718fcae93540511b7f5832f4e71fd835
python -m pip wheel --no-deps --no-build-isolation third_party/torchsparse_current -w runs/spvnas_fusion/wheels
python -m pip install --no-deps runs/spvnas_fusion/wheels/torchsparse-2.1.0-*.whl

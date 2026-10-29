# LIDF + SPVNAS

透明物体 RGB-D 深度补全：在 [LIDF](https://github.com/NVlabs/implicit_depth)
Stage 1 保留 PointNet，并联接入官方已搜索的 [SPVNAS](https://github.com/mit-han-lab/spvnas)
骨干，将点特征汇聚到 LIDF 体素后做残差融合。

本仓库的模型、训练器与数据代码来自 **20G、epoch 57 成绩对应的恢复训练版本**。
发布时整理了路径、环境、验证与启动入口，没有替换模型、修改损失或重新训练。
20G/65G 表示 SPVNAS 官方模型档位，不表示完整深度补全系统的显存或实测计算量。

## 已核实的 20G 结果

使用同一个 epoch 57 checkpoint，逐图指标在每个集合内等权平均。
RMSE、MAE 单位为米；δ 指标单位为百分比，阈值为 1.05、1.10、1.25。

| 数据集 | 图像数 | RMSE | REL | MAE | δ1.05 | δ1.10 | δ1.25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| synthetic-val | 532 | 0.014295 | 0.017124 | 0.009313 | 92.98 | 97.13 | 99.45 |
| synthetic-test | 408 | 0.036307 | 0.053871 | 0.029760 | 60.40 | 84.58 | 98.92 |
| **real-val** | **173** | **0.035802** | **0.045657** | **0.027262** | **69.23** | **89.82** | **97.61** |
| real-test | 113 | 0.049664 | 0.080815 | 0.044082 | 33.44 | 61.35 | 97.21 |

- 完整训练累计 60 epoch；该 checkpoint 已完成 57 epoch、80,940 次参数更新。
- epoch 57 是 **13 个保留权重中，real-val 六指标等权平均名次最佳**。
  旧版仅按 real-val RMSE 选出的 epoch 37 是不同选择规则的结果。
- real-val 参与了选模。没有完成预算相同的 60-epoch original 对照，不能据此断言融合带来提升。
- 这是 Stage 1 输出；不与论文 Stage 2 refinement 后的结果混为一谈。
- 原始模型输入 320×240；指标在 256×144 上按透明物体区域及有效 GT 口径计算。
- 完整数字、checkpoint SHA-256 与选模范围见 [provenance/epoch57.json](provenance/epoch57.json)。

## 代码结构

```text
src/models/                 LIDF、SPVNAS 残差分支、稳定射线选择
src/trainers/train_spvnas.py 四卡 DDP、梯度累积、学习率分组、完整续训
src/datasets/               ClearGrasp 与缓存像素网格的预处理实现
src/utils/training_resume.py 模型/Adam/scheduler/各 rank RNG 恢复与协议检查
src/utils/ipc_runtime.py    临时目录、DataLoader 超时和各 rank 错误协调
src/extensions/             LIDF ray–AABB、point–AABB CUDA 源码
src/experiments/            original、20G、65G、冻结/随机初始化配置
scripts/spvnas/             发布版训练、验证、评估和计时入口
configs/                   60-epoch 配置、续训示例、固定搜索结构
third_party/spvnas/         官方固定提交的模型源码与许可证
archive/                   原服务器脚本与整理前入口，供历史追溯
```

SPVNAS 输入为有效输入深度点云与 RGB，经 `Linear(6,4)` 适配；使用 0.05 m
稀疏网格、批次独立量化和逆映射，再按 LIDF 体素索引均值汇聚。
20G/65G 分类前特征分别为 48/76 维，残差适配到原 PointNet 的 128 维。
新分支不读取 GT 深度。官方骨干保持固定搜索结构，不执行架构搜索。

## 环境与安装

已使用的环境：Linux、Python 3.10.20、PyTorch 2.1.2+cu121、torchvision 0.16.2+cu121、
CUDA toolkit 12.1、RTX 3090；每个正式实验四张卡，每卡 24 GB。
NumPy 1.23.5、torch-scatter 2.1.2+pt21cu121、torchpack 0.3.1。

TorchSparse 的版本字符串为 2.1.0，但必须使用提交
`385f5ce8718fcae93540511b7f5832f4e71fd835`；旧 `v2.1.0` 标签缺少所需接口。

```bash
git clone https://github.com/qucheng-wuan/LIDF-SPVNAS.git
cd LIDF-SPVNAS
conda create -n lidf-spvnas python=3.10 -y
conda activate lidf-spvnas

python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install torch-scatter==2.1.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

还需要 C++ 编译工具、CUDA 12.1 toolkit 和 OpenEXR 系统依赖。
Ubuntu 可安装 `build-essential libopenexr-dev`；CUDA toolkit 使用已有安装，
或在 conda 环境通过 NVIDIA channel 安装 `cuda-toolkit=12.1`。
PyTorch CUDA wheel 本身不包含完整的 `nvcc` 编译工具链。

```bash
# 替换为本机 CUDA 12.1 toolkit；若装在 conda 环境内可用 "$CONDA_PREFIX"。
export CUDA_HOME=/usr/local/cuda-12.1
export TORCH_CUDA_ARCH_LIST=8.6  # RTX 3090；其他 GPU 按实际架构修改
bash scripts/install_sparse_dependencies.sh
python scripts/spvnas/fetch_weights.py
```

安装脚本下载并固定 TorchSparse 与 sparsehash 提交；没有要求安装 numba 或 mpi4py。
LIDF 自身的两个 CUDA 扩展会在首次导入时 JIT 编译。RGB ResNet 使用原版的
ImageNet 初始化，首次构建需要网络下载或已有 torch hub 缓存。

SPVNAS 下载器使用历史官方主机 `hanlab18.mit.edu`，并验证
[固定 SHA-256](provenance/official_assets.json)。若远端不可用，可把同哈希的
`net.config` 与 `init` 放到下列目录，或设置 `SPVNAS_ASSETS_DIR` 指向同样结构的缓存：

```text
runs/spvnas_fusion/assets/
├── SemanticKITTI_val_SPVNAS@20GMACs/{net.config,init}
└── SemanticKITTI_val_SPVNAS@65GMACs/{net.config,init}
```

## 数据

```text
/path/to/cleargrasp-root/
├── cleargrasp-dataset-train/
└── cleargrasp-dataset-test-val/
    ├── synthetic-val/
    ├── synthetic-test/
    ├── real-val/
    └── real-test/
```

```bash
export LIDF_DATA_ROOT=/path/to/cleargrasp-root
```

根路径应包含这两个顶层子目录。数据不随代码发布。对应训练使用 45,454 张
ClearGrasp 合成图像，不混用 Omniverse。保留原版合成输入深度损坏与增强流程。
YAML 中相对路径按仓库根目录解析；无需修改源码中的服务器路径。

## 验证与训练

先下载两种模型的官方资产，再运行验证；下载资产或改动源码/配置后旧验证签名会失效。
用 `nvidia-smi` 选择空闲卡。下面 GPU 编号仅为示例。

```bash
python -m unittest discover -s tests -v
python scripts/spvnas/test_mapping.py
python scripts/spvnas/test_ray_selection.py

# 真实数据：forward、batch 隔离、无 GT 推理、20 步 backward、checkpoint 回读。
python scripts/spvnas/validate.py spvnas_fusion_20g --gpu 0 --overlay configs/train_60ep.yaml

# 四卡 × micro-batch 2 × 累积 4 = 有效 batch 32。
python scripts/spvnas/launch.py spvnas_fusion_20g --gpus 0,1,2,3 --port 29520 \
  --overlay configs/train_60ep.yaml --overlay configs/20g_60ep.yaml
```

`launch.py` 要求当前变体具有匹配源码、实际资产及有效配置的验证记录，检查 GPU 占用，
并拒绝覆盖已有输出目录。前台进程保持运行；长期作业可放在自己管理的 tmux 会话中。

`original` 与 `spvnas_fusion_65g` 使用同样的验证/启动命令，替换变体名称及输出 overlay
为 `configs/original_60ep.yaml` / `configs/65g_60ep.yaml`。
支持 `freeze_*.yaml`、`random_*.yaml` 消融；冻结实验需提供兼容的 Stage 1 初始化权重，
并使用 `validate.py --frozen` 验证。冻结/随机配置不代表已经完成对应正式对照训练。

新输入适配/融合层 lr=1e-3；原 LIDF 和 SPVNAS 骨干 lr=1e-4。
`configs/train_60ep.yaml` 明确使用 epoch 预算，覆盖早期配置的 10,000-step 上限。
60 epoch 约 85,200 次更新。训练中按 synthetic-val RMSE 保存 best；
后处理的 real-val 六指标选模是独立步骤。

## 完整断点续训

仅支持完整 epoch 边界续训，恢复模型、Adam、scheduler、步数、训练协议以及各 rank
Python/NumPy/PyTorch/DataLoader 随机状态。必须提供同一次运行的 latest 与历史 best，
并使用新的输出目录；GPU 数和训练协议必须一致。

```bash
python scripts/spvnas/launch.py spvnas_fusion_20g --gpus 0,1,2,3 --port 29520 \
  --overlay configs/train_60ep.yaml \
  --resume runs/spvnas_fusion/20g_60ep/ckpt/stage1/latest_network.pth \
  --resume-best runs/spvnas_fusion/20g_60ep/ckpt/stage1/best_network.pth \
  --output-dir runs/spvnas_fusion/20g_resumed
```

该 epoch-57 版本保留 `latest_network.pth` 与 `best_network.pth`，不自动永久保留每一轮。
需要逐 epoch 比较时应另行归档权重。移动数据后路径/数据身份检查可能拒绝历史续训；
不会静默跳过一致性检查。缺失完整 RNG 的旧权重不能保证无缝复现。

## 评估与计时

训练后的 epoch 57 权重未包含在 Git 中。其文件大小为 301,033,012 字节，SHA-256：

```text
c9abcf490a20db65b54d4bda08681db41a546808b2801d96b24beb04607ce6de
```

准备对应权重、官方 SPVNAS 资产及数据后：

```bash
python scripts/spvnas/eval_checkpoint.py spvnas_fusion_20g --gpu 0 \
  --checkpoint /path/to/epoch_57.pth --overlay configs/train_60ep.yaml \
  --output-dir runs/eval_epoch57

python scripts/spvnas/benchmark.py spvnas_fusion_20g --gpu 0 \
  --checkpoint /path/to/epoch_57.pth --overlay configs/train_60ep.yaml \
  --output runs/benchmark_epoch57/result.json
```

评估默认跑四个集合；可用 `--sets real-val` 单独评估。输出目录必须不存在。
保留逐图 CSV、汇总、运行配置、源码签名与 checkpoint 哈希。
`eval_all_sets.py` 为同一入口的兼容名称。计时为单个固定 synthetic-val 样本，
10 次预热、50 次计时，包含传输、几何、两路特征和预测头；不是数据集总体吞吐量。

历史代码导出的 `log10` 列实际使用自然对数。为了保留历史指标口径，该实现未在发布时
静默更改；上表的六项指标不包含此列。原训练数据加载、模型权重与状态可以核查，
但共享 GPU、浮点归约及历史续训过程意味着从头重训不承诺逐位相同结果。

## 许可证与来源

请遵守 [NVIDIA 非商业研究/评估许可证](LICENSE)，保留原作者署名。
第三方来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，
上游说明见 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)。

```bibtex
@inproceedings{zhu2021rgbd,
  title={RGB-D Local Implicit Function for Depth Completion of Transparent Objects},
  author={Zhu, Luyang and Mousavian, Arsalan and Xiang, Yu and Mazhar, Hammad and
          van Eenbergen, Jozef and Debnath, Shoubhik and Fox, Dieter},
  booktitle={CVPR}, year={2021}
}
@inproceedings{tang2020searching,
  title={Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution},
  author={Tang, Haotian and Liu, Zhijian and Zhao, Shengyu and Lin, Yujun and
          Lin, Ji and Wang, Hanrui and Han, Song},
  booktitle={ECCV}, year={2020}
}
```

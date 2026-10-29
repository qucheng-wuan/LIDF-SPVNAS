# Sources and licenses

This repository derives from [NVlabs/implicit_depth](https://github.com/NVlabs/implicit_depth),
commit `4dc85bbddfc96af60006d6006be0c74478fe9204`. Its NVIDIA Source Code License
requires non-commercial research/evaluation use. The complete license is retained
in [LICENSE](LICENSE); original copyright and attribution notices are preserved.

The SPVNAS source subset in `third_party/spvnas` is unchanged from
[mit-han-lab/spvnas](https://github.com/mit-han-lab/spvnas), commit
`5c53224ca685844eb27caedf22817a89233ac0ce`. It includes the official model zoo,
core models/modules and configuration files. Its [MIT license](third_party/spvnas/LICENSE)
is retained. The integration patches point-to-voxel behavior at runtime in
`src/models/spvnas_fusion.py`; the vendored source itself is unchanged.

The installer fetches TorchSparse commit
`385f5ce8718fcae93540511b7f5832f4e71fd835` from
[mit-han-lab/torchsparse](https://github.com/mit-han-lab/torchsparse) (MIT),
and sparsehash commit `f93c0c69e959c1c77611d0ba8d107aa971338811` from
[sparsehash/sparsehash](https://github.com/sparsehash/sparsehash) (BSD-style).
Their license files remain in the fetched repositories. The TorchSparse version
string is 2.1.0, but the older `v2.1.0` tag does not provide the same API.

Official SPVNAS model artifacts are downloaded separately from the legacy
Han Lab model host. They are not included in Git. SHA-256 digests and searched
network configurations are included for identification and integrity checks.
ClearGrasp data, ImageNet/ResNet pretrained weights and trained experiment
checkpoints are not redistributed in this code repository.

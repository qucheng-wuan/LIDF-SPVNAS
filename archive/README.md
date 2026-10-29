# Historical server scripts

`server_scripts/` and `original_entrypoints/` preserve the original Python/shell
scripts from the 20G continuation/recovery experiment and its evaluation.
They contain historical absolute paths, GPU assignments and process-control
assumptions. They are provenance references, not the documented release entrypoints.
Use the commands in the root README for training, validation, resume and evaluation.

The model, geometry, losses and training implementation in `src/` come from the
epoch-57 source tree. The active `scripts/spvnas/` entrypoints add portable paths,
process environment initialization, configuration-bound validation, and separate
evaluation output directories. No historical successful validation is presented
as validation of newly adapted launch code.

The 65G eight-GPU migration was an experiment and never became the formal training
run. It is not part of this release. Both 20G and 65G architecture configurations
remain available through the common four-GPU training implementation.

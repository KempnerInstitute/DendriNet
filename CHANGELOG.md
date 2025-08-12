 -- November 2024 --> main codebase strcuture including EI layers and dendrinet
 -- February 2025 --> analysis and branch ablation, and information tools
 -- March 2025 --> pre-factor version --> v0.1.0
 -- July 2025  --> first refactor version --> v1.0.0
    Clean codebase structure
    added slurm generation for fsdp and distributed training
    added new sweep workflow
-- August 2025 --> added regularization and prunning
    enhanced information analysis and training optimization
    Added computation_level parameter (branch/layer/network)
    Added multivariate aggregation option for information analysis
    Added configurable training optimization parameters (NaN detection, adaptive clipping)
    Added hash-based encoder network caching
    Added encoder_net_trainer/main_trainer

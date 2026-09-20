# Matched 8,000-update discount ablation

Actual Python experiment comparing gamma 0 against the tuned gamma 0.99 benchmark. Both conditions use seeds 11, 23, 37, 51 and 71 on fixed candidate-grouped partitions. All neural models use 8,000 optimizer updates, learning rate .001 and batch size 64. The original source-validation selection is inherited unchanged; gamma zero is not independently tuned.

Architecture, input features, preprocessing, data/replay/exploration seeds, reward weighting and supervised control settings remain matched. Thresholds are selected separately on source validation and then frozen for both tests. Thus threshold-dependent changes include source-threshold adaptation; AP provides a threshold-independent comparison.

Entries are mean +/- sample SD across five training seeds. Paired changes are gamma 0 minus gamma 0.99. No significance or convergence claim is made.


## Findings

Removing the future-value term slightly improved mean within-dataset AP: DDQN changed from 0.9690 to 0.9704 on ISCX and from 0.8876 to 0.8972 on Mendeley. Both remain below the unchanged tuned Random Forest AP (0.9759 and 0.9135).

Transfer is mixed. DDQN AP fell from 0.3953 to 0.3784 for ISCX to Mendeley, but rose from 0.5016 to 0.5958 in reverse. At the source-validation 1% FPR operating point, reverse-transfer FPR fell from 45.04% to 18.25%, with recall falling from 28.66% to 27.12%. ISCX-to-Mendeley FPR remained approximately 96.24%. These thresholds do not meet a 1% target FPR constraint.

The matched longer-budget experiment supplies no consistent evidence that gamma 0.99 improves this formulation. It does not prove gamma zero is universally preferable. At gamma zero, the future-value term vanishes, and the identical DQN/DDQN results are expected from their otherwise matched implementation. This experiment provides no DDQN-specific advantage. The row-to-next-row transitions remain independent of the selected action and do not establish a realistic sequential decision task.

All 360 metric rows and source-validation thresholds passed saved-prediction verification; 180 prediction files were paired by sample identity. All 29 automated tests passed, including safeguards against unintended configuration changes. Metric verification uses the same metric routines as training, so it establishes artifact consistency rather than independent formula validation. These findings were verified on 20 September 2026.

## default

| Source | Test | Model | Metric | Gamma .99 | Gamma 0 | Paired change |
|---|---|---|---|---:|---:|---:|
| iscx | iscx | ddqn | average_precision | 0.9690 +/- 0.0003 | 0.9704 +/- 0.0005 | 0.0014 +/- 0.0007 |
| iscx | iscx | ddqn | f1 | 0.9047 +/- 0.0040 | 0.9083 +/- 0.0018 | 0.0037 +/- 0.0048 |
| iscx | iscx | dqn | average_precision | 0.9693 +/- 0.0003 | 0.9704 +/- 0.0005 | 0.0011 +/- 0.0006 |
| iscx | iscx | dqn | f1 | 0.9057 +/- 0.0017 | 0.9083 +/- 0.0018 | 0.0026 +/- 0.0035 |
| iscx | mendeley | ddqn | average_precision | 0.3953 +/- 0.0417 | 0.3784 +/- 0.0336 | -0.0169 +/- 0.0563 |
| iscx | mendeley | ddqn | f1 | 0.6375 +/- 0.0055 | 0.6370 +/- 0.0044 | -0.0006 +/- 0.0048 |
| iscx | mendeley | dqn | average_precision | 0.4037 +/- 0.0516 | 0.3784 +/- 0.0336 | -0.0253 +/- 0.0315 |
| iscx | mendeley | dqn | f1 | 0.6395 +/- 0.0062 | 0.6370 +/- 0.0044 | -0.0025 +/- 0.0039 |
| mendeley | iscx | ddqn | average_precision | 0.5016 +/- 0.0728 | 0.5958 +/- 0.0516 | 0.0941 +/- 0.0965 |
| mendeley | iscx | ddqn | f1 | 0.6609 +/- 0.0001 | 0.6610 +/- 0.0000 | 0.0001 +/- 0.0001 |
| mendeley | iscx | dqn | average_precision | 0.5317 +/- 0.0821 | 0.5958 +/- 0.0516 | 0.0641 +/- 0.1254 |
| mendeley | iscx | dqn | f1 | 0.6610 +/- 0.0000 | 0.6610 +/- 0.0000 | 0.0000 +/- 0.0000 |
| mendeley | mendeley | ddqn | average_precision | 0.8876 +/- 0.0026 | 0.8972 +/- 0.0044 | 0.0096 +/- 0.0052 |
| mendeley | mendeley | ddqn | f1 | 0.8566 +/- 0.0004 | 0.8585 +/- 0.0023 | 0.0019 +/- 0.0021 |
| mendeley | mendeley | dqn | average_precision | 0.8879 +/- 0.0032 | 0.8972 +/- 0.0044 | 0.0093 +/- 0.0059 |
| mendeley | mendeley | dqn | f1 | 0.8570 +/- 0.0008 | 0.8585 +/- 0.0023 | 0.0015 +/- 0.0019 |


## validation_fpr_limit

| Source | Test | Model | Metric | Gamma .99 | Gamma 0 | Paired change |
|---|---|---|---|---:|---:|---:|
| iscx | iscx | ddqn | recall | 0.6986 +/- 0.0070 | 0.6947 +/- 0.0333 | -0.0040 +/- 0.0338 |
| iscx | iscx | ddqn | fpr | 0.0084 +/- 0.0009 | 0.0071 +/- 0.0036 | -0.0013 +/- 0.0029 |
| iscx | iscx | dqn | recall | 0.6926 +/- 0.0106 | 0.6947 +/- 0.0333 | 0.0021 +/- 0.0431 |
| iscx | iscx | dqn | fpr | 0.0086 +/- 0.0021 | 0.0071 +/- 0.0036 | -0.0015 +/- 0.0047 |
| iscx | mendeley | ddqn | recall | 0.7811 +/- 0.0067 | 0.7740 +/- 0.0088 | -0.0071 +/- 0.0151 |
| iscx | mendeley | ddqn | fpr | 0.9625 +/- 0.0006 | 0.9624 +/- 0.0011 | -0.0001 +/- 0.0014 |
| iscx | mendeley | dqn | recall | 0.7718 +/- 0.0065 | 0.7740 +/- 0.0088 | 0.0023 +/- 0.0136 |
| iscx | mendeley | dqn | fpr | 0.9623 +/- 0.0006 | 0.9624 +/- 0.0011 | 0.0000 +/- 0.0014 |
| mendeley | iscx | ddqn | recall | 0.2866 +/- 0.0557 | 0.2712 +/- 0.0835 | -0.0154 +/- 0.1145 |
| mendeley | iscx | ddqn | fpr | 0.4504 +/- 0.1998 | 0.1825 +/- 0.0829 | -0.2679 +/- 0.1967 |
| mendeley | iscx | dqn | recall | 0.2875 +/- 0.0403 | 0.2712 +/- 0.0835 | -0.0163 +/- 0.1118 |
| mendeley | iscx | dqn | fpr | 0.4004 +/- 0.2158 | 0.1825 +/- 0.0829 | -0.2179 +/- 0.2295 |
| mendeley | mendeley | ddqn | recall | 0.1805 +/- 0.0104 | 0.1974 +/- 0.0154 | 0.0169 +/- 0.0253 |
| mendeley | mendeley | ddqn | fpr | 0.0107 +/- 0.0021 | 0.0105 +/- 0.0010 | -0.0001 +/- 0.0023 |
| mendeley | mendeley | dqn | recall | 0.1830 +/- 0.0128 | 0.1974 +/- 0.0154 | 0.0145 +/- 0.0273 |
| mendeley | mendeley | dqn | fpr | 0.0107 +/- 0.0023 | 0.0105 +/- 0.0010 | -0.0002 +/- 0.0025 |


## Control checks

Matched 180 validation/test prediction files by sample identity and label. Checked 120 non-RL control files to absolute score tolerance 1e-12; maximum score difference 3.33e-16; changed operating-point decisions 0. DQN/DDQN predictions and training histories at gamma zero match exactly in 40 files. Actual neural optimizer updates and sample exposures match the tuned reference.


## Evidence and limitations

The [aggregate evidence](../research/results/gamma-zero-tuned/) includes every per-seed metric, mean/sample-SD summaries, paired changes, configuration and verification records. Local predictions, checkpoints, training histories and curve points are indexed in the [artifact inventory](../research/results/artifact-inventory.json). Original datasets are unchanged and remain local.

This isolates gamma under inherited gamma-.99-selected settings; it does not compare independently optimized algorithms. Seed variation covers training randomness on fixed partitions. Prior target inspection informed the redesign, feature extraction equivalence is not fully established, and raw URL/domain identifiers are unavailable. The 8,000-update boundary was selected by earlier bounded tuning; completing this run does not establish convergence.


## Reproduction

Use the recorded Python environment and prepared exports, from the repository root. Keep completed output directories; use new directories for any rerun.

```text
python scripts/run_research_benchmark.py --prepared work/prepared-candidate-v2 --output work/ablation-gamma-zero-tuned-v1 --purpose exploratory --seeds 11 23 37 51 71 --tuning-selection work/tuning-source-v1/selection.json --discount-ablation-from-tuning --gamma 0 --trees 200 --threads 2
python scripts/verify_benchmark.py --run work/ablation-gamma-zero-tuned-v1 --output work/verified-gamma-zero-tuned-v1
python scripts/compare_gamma_ablation.py --baseline work/benchmark-tuned-v1 --ablation work/ablation-gamma-zero-tuned-v1 --output work/compared-gamma-zero-tuned-v1
python scripts/report_long_gamma_ablation.py --baseline work/benchmark-tuned-v1 --ablation work/ablation-gamma-zero-tuned-v1 --comparison work/compared-gamma-zero-tuned-v1 --output docs/ablation-gamma-tuned-results.md
python scripts/record_experiment_artifacts.py --work work --output research/results/artifact-inventory.json
```

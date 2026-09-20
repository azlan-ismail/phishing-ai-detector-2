# Experiment register

This register separates implementation checks from exploratory scientific comparisons. None of the runs establishes a final confirmatory manuscript result. All main five-seed conditions use the same candidate-grouped 60/20/20 partitions. Source-fitted transformations and raw input hashes are documented in [preprocessing](preprocessing.md).

| Stage | Local run directory under work/ | Settings | Record |
|---|---|---|---|
| Workflow smoke test | benchmark-smoke-v1 | Seed 11; 250 neural updates; 50 RF trees | [Training](training.md), research/audits/training-smoke-validation.json |
| Source-validation budget pilot | budget-pilot-v1 | Seed 101; 1,000/2,000/4,000 updates; 18 neural fits; selected 1,000 | [Budget policy and results](../research/results/five-seed/) |
| Four-feature baseline | benchmark-five-v1 | Seeds 11,23,37,51,71; gamma .99; 1,000 updates; 200 RF trees | [Five-seed results](five-seed-results.md) |
| Remove URL length and ratio | ablation-length-v1 | Same five seeds/budget; gamma .99; excluded inputs and indicators zeroed | [Feature ablation](ablation-length-results.md) |
| Bounded source-only tuning | tuning-source-v1 | Seed 101; six candidates per trained model/source; 60 candidate evaluations | [Tuning report](tuned-source-results.md) |
| Tuned five-seed evaluation | benchmark-tuned-v1 | Same reporting seeds; selected neural budget 8,000, learning rate .001; selected LR/RF settings | [Tuning report](tuned-source-results.md) |
| Remove future-value target term | ablation-gamma-zero-v1 | Same five seeds/budget; four features; gamma 0 | [Discount ablation](ablation-gamma-results.md) |

Each of the four main conditions contains 60 model/source/seed configurations (including trivial controls), 120 source-target evaluations, and 360 operating-point metric rows. Together these are 240 configurations, 480 evaluations and 1,440 metric rows. Repeated controls are not independent evidence or additional datasets. The original pilot's 18 fits, new tuning stage's 60 candidate evaluations (12 neural trajectories plus 24 supervised fits), and smoke's 12 configurations are separate.

## Stored evidence

- Public per-seed aggregate metrics include source, target, model, seed, operating point, threshold, confusion counts, discrimination metrics, predicted-positive proportion and timing.
- Public summaries contain mean, sample SD and count across seeds. Paired feature/discount changes retain each matched seed. No significance tests are claimed.
- Public verification records identify the exact software/configuration and preparation/script hashes. Comparison verification checks paired sample identities, controls and RL equivalence where applicable.
- Local run folders retain validation/test scores and predictions, checkpoints, training histories, PR/ROC curve points and threshold-selection details.
- [Artifact inventory](../research/results/artifact-inventory.json) lists relative filenames, byte sizes and SHA-256 hashes for completed training/pilot/verification/comparison folders. It records local integrity, not remote availability.
- Original user CSVs remain unchanged and are not uploaded. Raw hashes and provenance are recorded in the [feature audit](feature-compatibility.md).

## Interpretation rules

Budget selection used source validation at gamma .99. Both ablations inherit 1,000 updates rather than retuning for their condition. Each model selects its own source-validation thresholds, frozen before both test evaluations. Seed SD covers training randomness on fixed partitions, not uncertainty across independently sampled datasets.

The feature-removal hypothesis followed whole-table and target-result inspection; treat it as post-hoc. The gamma experiment is a fixed-setting diagnostic, not proof of optimal gamma. Shared feature definitions remain only partly verified; the ISCX rename is a user-provided assumption. The Mendeley published convention is confirmed as 0 legitimate, 1 phishing.

No single run should replace earlier results silently. Further conditions must use new output directories and append a new register entry. Bounded tuning is now complete and verified. All neural models selected the largest tested budget, so convergence remains unestablished. Independent confirmation, broader source-only tuning, a matched longer-budget gamma comparison, and reward/resampling ablations remain open.

## Latest verified stage (20 September 2026)

The tuned stage verified all 360 metrics, 60 candidate objectives and 180 paired prediction files. All 26 tests passed. The inventory now covers 2,615 files in 14 experiment/verification folders. Per-seed aggregates, sample SDs, paired changes and source-validation learning curves are preserved under research/results/tuned-source. Random Forest leads within-dataset AP; cross-dataset limitations and lack of consistent DDQN advantage remain. Earlier results are retained.

# Source-only validation plateau diagnostic

Six actual Python training trajectories completed using the original pilot seed 101. This stage evaluated source validation only: no test exports were opened and no test metrics were produced. It extends the earlier gamma-0.99 tuning pilot without replacing any five-seed results.

## Rule fixed before training

MLP, DQN and DDQN retained learning rate .001, batch size 64, unweighted training, the existing architecture, source-fitted preprocessing and fixed partitions. RL gamma remained .99. Each trajectory was rerun from its original initialization, preserving optimizer, replay and RNG evolution; restarting from weights alone was avoided. All six 8,000-update validation score vectors exactly reproduced the previous pilot checkpoints.

Validation AP was checked at 8,000, 12,000, 16,000, 20,000, 24,000, 28,000 and 32,000 updates as needed. The initial checkpoint established the reference score. An improvement of at least .001 above the last meaningful best reset the counter; otherwise the counter increased. Two consecutive failures stopped training, subject to a maximum of 32,000 updates. Small gains could accumulate relative to the reference before resetting it.

The selected checkpoint maximized observed validation AP, with exact ties choosing the earliest checkpoint. This selection rule differs from the meaningful-gain rule used to stop training. The saved policy was written before the trajectories started.

## Results

| Source | Model | AP at 8,000 | Best validation AP | Gain | Selected updates | Stopped at |
|---|---|---:|---:|---:|---:|---:|
| ISCX | MLP | 0.973411 | 0.973963 | +0.000552 | 16,000 | 16,000 |
| ISCX | DQN | 0.969937 | 0.971176 | +0.001239 | 12,000 | 20,000 |
| ISCX | DDQN | 0.969625 | 0.970050 | +0.000425 | 16,000 | 16,000 |
| Mendeley | MLP | 0.911828 | 0.911828 | 0.000000 | 8,000 | 16,000 |
| Mendeley | DQN | 0.898883 | 0.905708 | +0.006825 | 24,000 | 28,000 |
| Mendeley | DDQN | 0.901037 | 0.906999 | +0.005961 | 28,000 | 32,000 |

All six met the operational plateau rule. Mendeley DDQN met it at the budget ceiling, so its observed stopping point cannot distinguish a robust plateau from a temporary pause at the search boundary. Later improvement remains possible for every model.

ISCX gains were small, and Mendeley MLP did not improve beyond 8,000 updates. Mendeley DQN/DDQN gained approximately .0068/.0060 in validation AP with longer training. A single universal 8,000-update budget therefore did not capture the best observed source-validation checkpoint for every model. These are selection results on one reused pilot seed, not new test-performance estimates or evidence of DDQN superiority.

## Verification and records

All 32 automated tests passed, including the plateau counter and exact early-stop/prefix equivalence. Saved validation scores reproduced all 26 checkpoint AP values. Verification reproduced checkpoint choices and stopping decisions, checked uninterrupted history lengths and actual sample exposures, and checked the access record for the four prepared source train/validation exports. The runtime also asserted exact agreement with all six earlier 8,000-update score vectors. Verification uses the same AP library but separately recalculates the stopping decisions from saved scores.

There were six trajectories, 128,000 total optimizer updates, 26 validation checkpoint evaluations and zero test evaluations. [Public evidence](../research/results/source-convergence/) contains the policy, per-checkpoint curves, results, software and hashes, and verification. Local checkpoints, validation scores and complete training histories are included in the [artifact inventory](../research/results/artifact-inventory.json). Original datasets and all previous experiment outputs remain unchanged.

## Interpretation and next decision

This is a bounded validation plateau diagnostic, not proof of mathematical convergence, convergence across seeds, or stability on independent samples. Source validation is reused from prior selection, and the overall research redesign followed earlier target inspection. Existing cross-dataset results remain exploratory because extraction equivalence is unresolved; this stage requires no RA response.

The selected budgets are candidates for a future frozen evaluation policy. Do not substitute these validation AP values into manuscript test-result tables. Before any further five-seed test evaluation, specify whether models receive their individually selected budgets (a comparison of tuned procedures) or a matched per-source update budget (a controlled algorithm comparison). DQN and DDQN now have different selected budgets, so those interpretations must not be conflated. No new five-seed budget change or test evaluation was performed here.

## Reproduction

Use the existing prepared exports and original tuning pilot, with the recorded Python environment. Use fresh output directories for reruns.

```text
python scripts/check_source_convergence.py --prepared work/prepared-candidate-v2 --reference work/tuning-source-v1 --output work/source-convergence-v1
python scripts/verify_source_convergence.py --run work/source-convergence-v1 --output work/verified-source-convergence-v1
python scripts/record_experiment_artifacts.py --work work --output research/results/artifact-inventory.json
```

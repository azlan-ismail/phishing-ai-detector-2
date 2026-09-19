# Reproducible preprocessing foundation

**Provenance update (19 September 2026):** Mendeley's published labels are now confirmed as 0 = legitimate and 1 = phishing. The ISCX file is treated as the IEEE file renamed without content changes, per the user's explicit assumption; this has not been independently byte-verified. Source papers support the four intended feature meanings, but exact extraction equivalence remains unverified. The [feature-compatibility audit](feature-compatibility.md) documents strong class-conditional distribution differences. This update supersedes earlier statements below that Mendeley's published label meaning is unresolved; historical results and manifests are unchanged.


Status: implemented and tested on the supplied original feature tables. These outputs are exploratory; source definitions and Mendeley numeric-label provenance remain unverified. No models have been trained with this pipeline.

## Inputs and scope

Provide local paths to ISCX_Phishing.csv and Mendeley_dataset.csv. The source tables have 15,367 rows / 79 predictors and 58,645 rows / 111 predictors respectively. They contain no raw URL/domain identifiers. Keep source CSVs local or in appropriately licensed storage; the public repository needs code and aggregate validation evidence only.

The script uses four candidate features in an explicit order:

| Candidate | ISCX input | Mendeley input |
| --- | --- | --- |
| URL length | urlLen | length_url |
| Domain length | domainlength | domain_length |
| Dot count in URL | NumberofDotsinURL | qty_dot_url |
| Domain/URL ratio | Recomputed from the two lengths | Recomputed from the two lengths |

Similar names are not proof of equivalent parsing conventions. These are provisional candidates for a minimal diagnostic comparison, not a validated replacement for every original feature. Four missingness indicators are always appended, giving eight numeric model inputs. With the supplied files the indicators are all zero. The ratio is derived and adds no new raw information beyond the two lengths.

Do not train a four-feature baseline against a fourteen-feature RL model and attribute differences to algorithms. All models in a comparison must use the same representation. Stronger within-dataset experiments using richer native features remain separate future work.

Unsupported maximum-token estimates, directory-symbol sums, continuity proxies, and constant extension placeholders are excluded. The generic invalid-value policy only applies to these four nonnegative count/length candidates: negative counts, nonpositive lengths, and nonfinite values become missing before arithmetic. It must not be applied indiscriminately to other source fields where -1 may encode a legitimate category.

## Splits and transformations

- A seeded greedy allocator targets 60/20/20 class counts while keeping groups intact. It uses labels only for split stratification, never model scores.
- Default groups hash all original numeric predictors, excluding labels. Full feature duplicates, including records with conflicting labels, stay together. This is not domain-level splitting.
- The optional candidate grouping keeps identical reduced feature representations together. Use this as a declared sensitivity analysis, not a post-hoc way of selecting better results.
- Group indivisibility can change exact partition proportions; counts are saved.
- Fit medians and MinMax parameters on source training only. Save them as JSON and reuse them for source validation/test and target test.
- Do not clip target values to [0,1]. Values beyond source extrema remain visible.
- Missingness indicators have a fixed schema, and an entirely missing source-training feature causes a clear error.
- No SMOTE, class weighting, feature selection, threshold tuning, or model training occurs at this stage.
- IDs contain dataset name, raw-file SHA256, and zero-based source-row ordinal. They are stable for an unchanged file, not permanent URL identifiers.

## Run

Python with NumPy and pandas is sufficient. Exact executed versions are recorded in each manifest; the old tutorial dependency environment is not assumed to have been validated.

```bash
python scripts/prepare_research_data.py --iscx "/local/ISCX_Phishing.csv" --mendeley "/local/Mendeley_dataset.csv" --output work/prepared-full --group-by full
python scripts/prepare_research_data.py --iscx "/local/ISCX_Phishing.csv" --mendeley "/local/Mendeley_dataset.csv" --output work/prepared-candidate --group-by candidate
python -m unittest discover -s tests -v
```

Each output directory must be new. The script refuses to overwrite an existing run. It writes two partition manifests, two transformer JSON files, eight transformed CSVs, and one manifest with input/output checksums and software versions. Export names distinguish source, target, and split. The eight CSVs cover train/validation/test for both sources plus both transferred test sets.

When loading an export, select the explicit numeric feature order plus missingness indicators. Exclude sample_id and label from model inputs. For tuning CV, refit preprocessing inside every training fold from raw candidates; do not cross-validate on these already transformed matrices. These exports are for the outer train/validation/test experiment.

## Scientific limits and next work

The new split does not turn previously inspected data into a genuinely independent confirmation set. Existing dataset results already informed redesign. Neither grouping mode proves domain independence or resolves cross-source URL overlap. Hashing full records cannot recover unavailable URLs.

Prior supplied-code findings are confirmed: DQN trained on its internal validation rows; DDQN rewards depended on row position; preprocessing exports overwrote each other; and the transfer construction performed arithmetic on sentinel values. Replaying the feature calculations reproduces 3,526 test rows with directory-symbol count -17 and letter count 16, and 10,288 rows with a query ratio of (-1)/(-1)=1. These scripts avoid those engineered features.

Although the old preprocessing code fit before splitting, the relevant selected-feature extrema and median happen to match training-only values in the supplied split. This observation does not establish inflation attributable to those statistics. The other implementation defects remain separate.

Next: verify source definitions/label provenance, obtain the balancing script, implement a shared DQN/DDQN training environment with label-based rewards and fixed seeds, and run matched supervised baselines. Do not interpret this preparation run as evidence of model performance.

# Cross-dataset validation: current evidence and next protocol

## Decision

Retain both ISCX-to-Mendeley and Mendeley-to-ISCX results as exploratory transfer evaluations. The saved preprocessing is reproducible and source-fitted. Identical upstream feature extraction and independent URL/domain identities are not established. No new model was trained or selected in this audit, and existing results remain unchanged.

## Verified on the actual files

The read-only Python audit reconstructed both source transformers from their original training rows, checked raw/partition/export hashes and label alignment, and reproduced both cross-dataset test exports within numerical tolerance. Neither transformer required fitting on target data. This verifies the implementation of the current numeric mapping, not the equivalence of the original extractors.

| Training source | Test source | Test rows | URL length outside source training range | Ratio outside source training range | Dot count outside source training range | Four-feature collisions with source training |
|---|---|---:|---:|---:|---:|---:|
| ISCX | Mendeley | 11,729 | 7,183 | 4,656 | 2 | 1,254 |
| Mendeley | ISCX | 3,073 | 0 | 0 | 0 | 1,104 |

Domain length had zero out-of-range values in both directions. Counts can overlap across features. Source min-max scaling deliberately does not clip target values; values outside [0,1] are expected when target measurements exceed source bounds. They are not evidence of a scaling implementation error, and target refitting or guessed clipping is not a justified repair. Being inside the source range does not establish distributional or semantic equivalence.

Collisions mean equal four-number feature vectors. They do not establish duplicate URLs or data leakage. The supplied tables contain numeric predictors, without raw URL/domain identity fields, so cross-source URL duplication and domain overlap cannot be checked from these files.

See [machine-readable audit](../research/audits/cross-dataset-protocol.json) for exact hashes and counts. This is post-hoc descriptive analysis of existing test data, not new independent validation.

## What the supplied legacy code establishes

The reviewed `Mendeley_Calc.py` copies `length_url` to `urlLen`, `domain_length` to `domainlength`, and `qty_dot_url` to `NumberofDotsinURL`; it derives the domain/URL ratio. Those operations support the current candidate field mapping but cannot establish how the original lengths or dot counts were extracted. Other legacy approximations, including average-based substitutes for longest-token features, are not used in the current four-feature benchmark.

The reviewed `check.py` fits imputation/scaling before splitting, and writes train and test outputs to the same filename. Those are historical issues; the current prepared exports instead use source-training-only preprocessing, as verified by reconstruction here. Neither supplied script is an original raw-URL feature extractor.

The [feature-compatibility audit](feature-compatibility.md) remains authoritative on known definitions and the class-conditional reversal in URL length/ratio. The accepted ISCX rename assumption is unchanged. Published Mendeley labels remain 0 legitimate and 1 phishing. No label reversal or guessed length adjustment is justified.

## Material needed to resolve the remaining uncertainty

Ask the RA for the following, with original filenames and dataset/version information:

1. Original extraction code for URL length, domain length and whole-URL dot count for each source, including dependency versions and any preprocessing before extraction.
2. If available, original URL strings with stable row identifiers linked to the numeric tables. A newly collected, unlinked URL list cannot verify the existing rows.
3. Explicit conventions for scheme inclusion, percent decoding, case/Unicode handling, missing schemes, host/subdomain parsing, ports, trailing dots, query strings and fragments.
4. Collection dates, original dataset versions, label provenance, and domain identifiers needed to assess overlap and temporal separation.

If raw URLs cannot be obtained, original extractors and their tests can strengthen semantic evidence, but row-level re-extraction and URL/domain-overlap verification will remain unavailable. Do not treat numeric columns alone as proof of harmonization.

## Protocol for a stronger follow-up

1. Preserve all existing runs as exploratory. Document the extraction conventions before new model fitting.
2. Where raw URLs are available, process both sources through one versioned extractor. Test its behavior on scheme, encoding, port, Unicode and path/query edge cases; do not visit the URLs to compute lexical features.
3. Record duplicate-URL, conflicting-label and registrable-domain overlap checks. Define source partitions and cross-source exclusions before evaluating model outcomes. State clearly if chronological or domain-disjoint evaluation is impossible.
4. Fit imputation/scaling and any balancing only on source training data. Select hyperparameters, stopping and thresholds on source validation. Freeze settings before target evaluation. Report source and target class prevalence.
5. Evaluate both directions with the same supervised/trivial references, per-seed AP/ROC-AUC and threshold-dependent precision, recall, F1, confusion counts and FPR. A source-validation 1% FPR rule does not guarantee 1% target FPR. Report default, source-F1 and source-FPR operating points rather than selecting the best on target labels.
6. Label any rerun on these already inspected datasets exploratory. For independent confirmation, acquire an untouched external collection with compatible extraction and freeze the protocol before inspecting its outcomes. Save all settings, hashes, predictions and aggregate results.

The immediate dependency is extraction evidence or row-linked raw URLs, not another run of the unchanged transfer experiment. Source-only convergence work can proceed separately while that evidence is obtained.

## Reproduction

```text
python scripts/audit_cross_dataset_protocol.py --prepared work/prepared-candidate-v2 --iscx path/to/ISCX_Phishing.csv --mendeley path/to/Mendeley_dataset.csv --output research/audits/cross-dataset-protocol.json
```

Audit completed on 20 September 2026. Original data, split assignments, training results and thresholds were not modified.

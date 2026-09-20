# Revision decisions and reviewer coverage

The [replacement manuscript draft](manuscript-revision.md) rewrites the title, abstract, introduction, scope of related work, methods, results, discussion, conclusion and availability statement around the verified controlled experiments. It is a substantive replacement draft, not an edited Springer PDF or a submission-ready typeset article. Authorship and author declarations require the authors' review and were not invented.

## Central decision

The paper now presents an empirical comparison of Q-learning and supervised methods on a restricted four-feature representation. It does not propose a new algorithm. A careful negative or mixed result can be informative, but this framing does not by itself resolve the associate editor's novelty objection. The authors must decide whether the evidence supports an empirical-study contribution appropriate to the eventual venue. No guarantee of novelty or acceptance is implied.

The original abstract's claims that the framework ensures robust cross-domain generalization, that DDQN consistently outperforms DQN, and that the results establish scalable resilience are removed. The title is narrowed to phishing classification because the supplied evaluated labels are binary benign/legitimate versus phishing. The former class-imbalance contribution is removed because the controlled redesign has no completed SMOTE/weighting comparison.

## Reviewer coverage

| Concern | Revision and evidence | Status / remaining limit |
|---|---|---|
| AE: minimal novelty | Reframe as an empirical audit, with explicit research questions and no new-method claim | Not resolved by writing alone; author/venue assessment required |
| AE and R1: fragmented writing, weak organization | Connected prose; conventional single-paragraph abstract; methods before results; fewer introductory definitions | Replacement draft complete; author language review remains |
| R1: artificial structured abstract | Replaced with a conventional paragraph | The reviewer quotes a structured version, while the supplied PDF extract has a different abstract; both sets of unsupported claims are avoided |
| R3: artificial Extension Letter Count | Constant extension placeholder excluded from the redesigned common representation | Does not constitute the requested isolated ablation of the old 14-feature model; old and new numbers must not be presented as a one-feature comparison |
| R3: operational false positives | Latest transfer FPR, precision, recall and mean false-positive counts reported; default all-phishing failure compared with trivial reference | Quantitatively addressed on these test tables; no operational deployment claim |
| R3: Gini threshold .01 | No Gini feature selection in revised primary pipeline; explicit common features instead | Original threshold sensitivity was not run; the original selection-benefit claim is removed |
| R3: feature selection/tuning/SMOTE/reward decomposition | Separate bounded tuning, feature-removal and gamma diagnostics; unweighted ±1 true-label rewards | Partial: gamma is not reward-shaping ablation; SMOTE/weighting and reward alternatives remain untested |
| R3: means/SD and significance | Five-seed means/sample SD in manuscript tables; paired differences archived | Descriptive uncertainty addressed; significance tests not performed and superiority/significance claims removed |
| R3: causal overestimation explanation | Causal claim removed; target equations and gamma-zero equality explained | No direct Q-overestimation measurement; discount ablation does not replace one |
| R3: unrealistic synthetic samples | No synthetic interpolation in controlled revision; scope excludes SMOTE benefits | Alternative imbalance strategies remain untested |
| R3: tuning harms transfer | Budget/selection stages separated; validation gains distinguished from test changes; thresholds and all-phishing failure discussed | Descriptive analysis, not proof of source overfitting or its cause |
| R3: calibration/adaptation and PR curves | Source-selected F1/FPR operating points and quantitative transfer burden included; curve points retained locally | No calibration/domain-adaptation experiment; publication-ready PR/ROC figures still to assemble |
| R3: literature engagement | Primary Double DQN/URLNet sources and recent DeepURLBench preprint consulted, with representation differences stated | Targeted update, not exhaustive current literature review; suggested intrusion-detection papers not added merely for citation count |
| R4: weak introduction and excessive basics | Research questions and empirical scope replace broad adaptation rhetoric; compact methods give actual environment and update rules | Replacement text complete |
| R4: imbalance handling | Class counts reported, no severe-imbalance or mitigation claim | New imbalance study would be needed to restore that contribution |
| R4: no new method/details | Explicit empirical-study framing and detailed implementation/protocol | Method detail improved; no new algorithm is claimed |
| AE: graphical presentation | Tables generated from verified per-seed files; misleading reward-to-overestimation story removed | Publication-quality figures are not yet delivered by this revision |

## How to integrate this draft

Replace the old headline results and associated narrative together. Do not combine old 14-feature/SMOTE results with new four-feature results in one purportedly matched comparison. Keep the 1,000-update feature ablation, 8,000-update gamma comparison and 16,000/28,000-update main comparison distinct, with their own reference conditions.

Preserve the distinction between a numeric mapping that reproduces correctly and proven extractor equivalence. The ISCX filename assumption is already accepted; no RA reply is required to retain the explicitly limited exploratory analysis. Independent confirmation and richer native-feature comparisons are not claimed as completed work.

Before submission, assemble figures from the recorded data: within-source AP by model with seed points; transfer PR curves with source-selected thresholds indicated; and source-validation learning curves showing observed checkpoints and the stopping rule. A seed-mean curve alone should not imply sampling confidence. Avoid target-optimized operating points. Saved curve data permit this work, but figure generation and visual QA remain outstanding.

The manuscript deliberately avoids an unverified novelty claim such as “first application of DDQN to malicious URL detection.” A broader literature check, author approval of the narrowed contribution, journal formatting, verified bibliography metadata, and author-specific declarations remain before submission. These are open items, not claims that the reviewer requests have all been satisfied.

## Evidence traceability

Main tables: `work/benchmark-matched-long-v1/metrics.csv` and the verified summary, publicly mirrored under `research/results/matched-long/`. Table-generation evidence is stored in `research/audits/manuscript-table-verification.json`. Historical diagnostic values are drawn from [feature ablation](ablation-length-results.md), [matched gamma ablation](ablation-gamma-tuned-results.md), [source-only plateau checks](source-convergence-results.md), and [cross-dataset audit](cross-dataset-validation.md). No models were retrained for this writing revision.

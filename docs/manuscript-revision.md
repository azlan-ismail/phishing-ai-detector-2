# Q-Learning and Supervised Baselines for Phishing URL Classification: A Controlled Four-Feature Study

## Abstract

Applying Q-learning to labelled URL records does not by itself establish an advantage over supervised classification or demonstrate adaptation to new threats. This study evaluates Deep Q-Network (DQN) and Double DQN (DDQN) alongside logistic regression, Random Forest, a supervised multilayer perceptron (MLP), and an always-phishing reference using two supplied phishing feature tables. All models receive the same four candidate features, with preprocessing fitted only on source training data. Five training seeds are evaluated on fixed within-source and cross-source partitions. Source-validation tuning and a bounded training-extension diagnostic inform matched neural budgets of 16,000 updates for ISCX and 28,000 for Mendeley. Random Forest obtains the highest within-source mean average precision, 0.9759 and 0.9135, compared with DDQN values of 0.9698 and 0.8873. MLP achieves the highest default-threshold mean F1 on Mendeley, illustrating that rankings depend on the metric. Cross-source performance remains weak: DDQN thresholds selected for at most 1% source-validation false-positive rate produce target false-positive rates of 96.41% and 36.70%. Separate matched-budget discount ablations reveal no consistent benefit from future-value bootstrapping. The findings do not support a consistent DDQN advantage in this formulation. Cross-source results remain exploratory because feature-extraction equivalence and URL/domain independence could not be established from the supplied numeric tables.

**Keywords:** phishing URL classification; DQN; Double DQN; supervised baselines; cross-dataset evaluation; reproducibility

## 1 Introduction

A classifier can perform well on held-out records from one collection while failing on another. Evaluating phishing detection therefore requires more than reporting a high within-dataset F1-score. The comparison must identify what information reaches each model, how training and thresholds are selected, and whether a transferred detector produces an acceptable false-positive burden. Differences in feature extraction can further complicate interpretation when the available datasets contain precomputed numeric records rather than original URL strings.

Q-learning provides one way to represent binary classification as action selection: a model receives a feature vector, chooses a class, and receives feedback derived from the label. However, an ordered stream of labelled records is not necessarily a sequential decision problem. If the selected action does not affect the next record, the benefits of a future-value target must be demonstrated rather than inferred from the use of a reinforcement-learning algorithm. Similarly, the overestimation motivation for Double DQN does not establish that overestimation explains differences observed in a particular URL experiment.

We examine three questions. First, do DQN and DDQN outperform supervised and trivial references when representation, preprocessing and evaluation conditions are shared? Second, do longer training and future-value bootstrapping provide consistent benefits under controlled comparisons? Third, what do cross-source results reveal about ranking quality, false-positive burden and the limits of the available feature mapping?

The contribution is an empirical audit and controlled comparison, rather than a new learning algorithm. The study records source-only model-selection decisions, separates training-budget and discount diagnostics, and preserves per-seed metrics and prediction-derived verification. Its scope is deliberately narrow: four candidate numeric features from two supplied binary phishing tables. The results cannot establish superiority over raw-URL models or general robustness to temporal drift.

## 2 Related work and scope

[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) introduced a separation between action selection and target evaluation to address overestimation in DQN, with evidence from Atari tasks. That result motivates comparison of the update rules, but it does not supply a causal explanation for classification differences in the present data. We therefore examine target construction directly through matched discount ablations and do not attribute outcomes to reduced overestimation without measuring it.

URL representation is another important distinction. [URLNet](https://arxiv.org/abs/1802.03162) learns representations from URL strings, whereas the present benchmark uses a small common set of precomputed numeric features. More recent work, [A New Dataset and Methodology for Malicious URL Classification](https://arxiv.org/abs/2501.00356), introduces DeepURLBench and explores string-based models augmented with DNS-derived information. These studies concern richer inputs and, in the latter case, a multiclass dataset. Their reported performance is not directly comparable with the binary, four-feature experiment here, and neither model was evaluated in this study.

The source resources also differ in scope. The [UNB URL-2016 description](https://www.unb.ca/cic/datasets/url-2016.html) covers several URL categories, while the supplied ISCX file used here contains benign and phishing records. The [Mendeley version-1 resource](https://data.mendeley.com/datasets/72ptz43s9v/1) provides phishing feature tables and states that label 0 denotes legitimate websites and label 1 denotes phishing. Our analysis concerns the supplied binary tables, not every category or dataset variant associated with these resources.

## 3 Data and preparation

### 3.1 Inputs and provenance

The supplied ISCX table contains 15,367 records with 79 predictors: 7,781 benign and 7,586 phishing. The Mendeley table contains 58,645 records with 111 predictors: 27,998 legitimate and 30,647 phishing. These distributions are not a severe-imbalance benchmark. The controlled experiments preserve their class distributions and use neither SMOTE nor class-weighted rewards. Consequently, this study does not claim an improvement in imbalance handling.

The local ISCX filename is treated as a content-preserving rename of the file associated with IEEE DataPort DOI 10.21227/xngk-3p42. This is a provenance assumption, not an independently established byte-level match to the repository download. The supplied Mendeley dimensions and class counts agree with the published small variant, but that agreement also does not establish byte identity. Input hashes identify the files actually evaluated.

No original URL strings or domain identifiers are available in the supplied tables. We therefore cannot verify cross-source URL duplicates, enforce domain-disjoint partitions, reconstruct original extraction conventions, or establish chronological separation.

### 3.2 Shared representation

Each model receives URL length, domain length, whole-URL dot count, and the ratio of domain length to URL length. ISCX fields `urlLen`, `domainlength` and `NumberofDotsinURL` are mapped to Mendeley fields `length_url`, `domain_length` and `qty_dot_url`. The ratio is recomputed identically from the two lengths. Four fixed missingness indicators give eight input positions; all indicators are zero in these supplied data. The ratio is derived information, not an additional independent measurement.

The field names and published high-level descriptions support these candidate mappings, but do not prove equivalent scheme inclusion, decoding, Unicode handling, host parsing or normalization. The supplied conversion code copies these fields and calculates the ratio; it is not an original extractor. Unsupported token-length approximations and the constant extension-letter-count placeholder from the earlier manuscript pipeline are excluded. This redesign is a new four-feature experiment, not a one-variable extension-feature ablation of the original fourteen-feature system.

Identical four-feature vectors are grouped before a seeded allocation targeting 60/20/20 train/validation/test proportions. The resulting ISCX partitions contain 9,221/3,073/3,073 records; Mendeley contains 35,186/11,730/11,729. The split seed is 42. Groups remain intact, including feature-identical rows with different labels. These groups establish separation of the selected numeric representation within each dataset, not independence of URLs, domains or collection sources.

Medians and min-max scaling parameters are fitted only on source training rows and reused for source validation, source test and target test. Target values are not clipped. Reconstruction from the raw numeric tables reproduced both transformers and both transfer exports. For ISCX-to-Mendeley transfer, 7,183 of 11,729 target rows have URL lengths outside the source training range. This demonstrates range mismatch, but does not identify whether it arises from URL populations or extraction conventions.

## 4 Models and experimental protocol

### 4.1 Models and action-independent transitions

The comparison includes an always-phishing reference, logistic regression, Random Forest, MLP, DQN and DDQN. The three neural models use eight inputs, two 128-unit ReLU hidden layers and two outputs. MLP optimizes supervised cross-entropy. DQN and DDQN optimize squared error against their respective action-value targets. All use Adam with learning rate .001, batch size 64 and gradient clipping at norm 1.

For the Q-learning models, the state is a record's feature vector and the action is its predicted binary class. A correct action receives reward +1 and an incorrect action receives -1, based on the true label. Records are shuffled for each training traversal. The next state is the next record in that order, regardless of the chosen action; the last record is terminal. The sequence does not represent observed browsing behaviour or threat evolution. No online adaptation or temporal concept-drift experiment is performed.

For transition $(s,a,r,s',d)$, the DQN target is

$$y=r+\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a').$$

DDQN instead uses

$$y=r+\gamma(1-d)Q_{\theta^-}(s',\operatorname{argmax}_{a'}Q_\theta(s',a')).$$

The replay capacity is 10,000 transitions, and target weights are copied every 100 optimizer updates. Exploration uses $\epsilon_u=\max(0.05,0.995^u)$ with zero-based update index $u$. Separate seeded random streams control record order, exploration and replay sampling. These settings are matched between DQN and DDQN. At gamma zero, the future-value term vanishes, so differences attributable solely to DQN/DDQN target selection should disappear.

### 4.2 Selection and budgets

A bounded search evaluates six candidates per trained model and source using validation average precision (AP) and pilot seed 101. Logistic regression tests C values .001, .01, .1, 1, 10 and 100. Random Forest crosses depths 8, 16 and unlimited with minimum leaf sizes 1 and 5, using 200 trees. Neural models cross learning rates .0003 and .001 with checkpoints at 1,000, 4,000 and 8,000 updates. Highest validation AP selects the candidate; exact ties favour the earliest candidate. Candidate counts are equal, but computational costs are not.

The search selects C=100 for logistic regression and depth 8 for Random Forest, with minimum leaf sizes 5 on ISCX and 1 on Mendeley. All neural selections initially reach 8,000 updates at learning rate .001. A subsequent source-only diagnostic extends the same pilot trajectories, checking AP every 4,000 updates from 8,000 to a maximum of 32,000. Two checks without a gain of at least .001 above the last meaningful best stop training. All six trajectories meet this operational rule, with Mendeley DDQN doing so at the ceiling. This is not proof of convergence.

For the principal revised comparison, the largest selected checkpoint within each source is applied to all three neural models: 16,000 updates on ISCX and 28,000 on Mendeley. These budgets are frozen before the five-seed runs and do not use test scores for this budget choice. They match optimizer updates, rather than selecting separate stopping points for DQN and DDQN. Supervised non-neural settings remain as selected above.

### 4.3 Evaluation and uncertainty

Reporting seeds are 11, 23, 37, 51 and 71, distinct from the pilot seed. Each source-trained model is evaluated on its own test set and the other source's test set. All comparisons use the same fixed partitions. AP and ROC-AUC are calculated from continuous scores; precision, recall, F1, false-positive rate (FPR), balanced accuracy and confusion counts are recorded at three operating points.

The default rule uses class scores, choosing legitimate on an exact tie. A second threshold maximizes source-validation F1; ties select the highest threshold. A third maximizes validation recall subject to FPR at most .01, breaking ties by lower FPR and then higher threshold. Both selected thresholds are frozen before either test evaluation. A source-validation FPR constraint is not a target guarantee. Neural scores are output differences, not calibrated probabilities.

We report mean and sample standard deviation across training seeds, together with paired seed-level differences. These repetitions describe algorithmic randomness on fixed records, not uncertainty across independently sampled datasets. No statistical-significance claim is made. The experiments are exploratory because earlier target inspection informed the overall redesign, despite source-only selection within the recorded stages.

## 5 Results

### 5.1 Within-source comparison

Table 1 reports the latest matched-budget evaluation. Random Forest has the highest mean AP in both datasets. MLP has higher AP than either Q-learning model and the highest default F1 on Mendeley. The ranking therefore depends on whether the objective is score discrimination or a particular operating point. DDQN has slightly higher default F1 than DQN on ISCX but slightly lower AP on both sources; these small differences do not establish a consistent advantage.

**Table 1. Within-source results, mean ± sample SD across five seeds. F1 uses the default rule.**

| Model | ISCX AP | ISCX F1 | Mendeley AP | Mendeley F1 |
|---|---:|---:|---:|---:|
| Always phishing | 0.4937 ± 0.0000 | 0.6610 ± 0.0000 | 0.5226 ± 0.0000 | 0.6864 ± 0.0000 |
| Logistic regression | 0.9407 ± 0.0000 | 0.8366 ± 0.0000 | 0.8709 ± 0.0000 | 0.8425 ± 0.0000 |
| Random Forest | 0.9759 ± 0.0004 | 0.9179 ± 0.0048 | 0.9135 ± 0.0011 | 0.8587 ± 0.0007 |
| MLP | 0.9735 ± 0.0004 | 0.9115 ± 0.0066 | 0.9100 ± 0.0013 | 0.8634 ± 0.0018 |
| DQN | 0.9701 ± 0.0006 | 0.9047 ± 0.0064 | 0.8877 ± 0.0046 | 0.8579 ± 0.0016 |
| DDQN | 0.9698 ± 0.0006 | 0.9095 ± 0.0034 | 0.8873 ± 0.0030 | 0.8575 ± 0.0014 |

### 5.2 Longer training and validation selection

Relative to the 8,000-update reference, mean ISCX DDQN AP increases from .969042 to .969793, while Mendeley DDQN AP changes from .887556 to .887286. DQN shows the same broad pattern: .969299 to .970053 on ISCX and .887859 to .887654 on Mendeley. By contrast, Mendeley MLP AP increases from .899568 to .909980. The pilot's Mendeley Q-learning validation gains do not translate into improved mean test AP in the reporting runs. This observation does not isolate a causal explanation such as overfitting; it demonstrates why pilot-selection scores cannot substitute for evaluation results.

### 5.3 Cross-source behaviour and false-positive burden

Table 2 presents DDQN transfer results as an operating-point illustration; the supplementary aggregate files retain all six models and all thresholds. In the latest ISCX-to-Mendeley run, the source-FPR-selected threshold flags an average of 5,399.2 of 5,600 legitimate target records. In reverse transfer, it flags 571.0 of 1,556 legitimate records. Fractional counts are averages across five runs, not individual-record counts.

**Table 2. DDQN transfer results, mean ± sample SD across five seeds. The last three columns use the source-validation 1% FPR constraint.**

| Source → target | AP | Default F1 | Precision | Recall | FPR |
|---|---:|---:|---:|---:|---:|
| ISCX → MENDELEY | 0.4111 ± 0.0828 | 0.6385 ± 0.0060 | 0.4710 ± 0.0021 | 0.7843 ± 0.0067 | 0.9641 ± 0.0004 |
| MENDELEY → ISCX | 0.5508 ± 0.0474 | 0.6610 ± 0.0000 | 0.4746 ± 0.1664 | 0.2984 ± 0.0321 | 0.3670 ± 0.1724 |

These false-positive rates correspond to approximately 964 and 367 flags per 1,000 legitimate records if the respective measured rates were maintained. This is an arithmetic illustration, not a deployment forecast. Alert yield would also depend on prevalence and workflow costs, neither of which was studied operationally. Under default thresholds, every Mendeley-trained DQN/DDQN seed labels every ISCX test record as phishing, yielding F1 .6610, identical to the always-phishing reference. High recall or a moderate F1-score alone would obscure that failure.

### 5.4 Diagnostic ablations

At a separately matched 8,000-update budget, setting gamma to zero increases DDQN within-source AP from .9690 to .9704 on ISCX and from .8876 to .8972 on Mendeley. Transfer AP decreases in one direction and increases in the other. Gamma-zero DQN/DDQN validation predictions, test predictions and training histories agree exactly under matched seeds; non-RL controls are unchanged. This supplies no consistent evidence that future-value bootstrapping helps this formulation. It does not prove that gamma zero is universally preferable or directly measure Q-value overestimation.

An earlier 1,000-update feature diagnostic zeros URL length and its derived ratio while preserving the eight input positions, partitions and initial neural dimensions. Transfer AP improves for all trained models in both directions, but within-source AP decreases. For DDQN, reverse-transfer default F1 rises from .6609 to .7724, while within-Mendeley F1 falls from .8557 to .5399. Because the hypothesis followed inspection of target results and whole-table diagnostics, this is a post-hoc sensitivity analysis, not an independently validated feature-selection procedure. These ablations are compared with their own matched references; their budgets are not conflated with the latest 16,000/28,000-update stage.

## 6 Discussion and limitations

The main result is the absence of a consistent Q-learning advantage under the studied controls. The experiment supports an empirical comparison of update rules, representations and operating points; it does not support claims of a new adaptive detector, superior DDQN generalization, or deployment readiness. Since the action does not affect the next URL record, the sequential structure is imposed by training order. Future work on genuine sequential security decisions would require an environment in which actions affect observations, costs or outcomes, rather than simply relabelling a static classification problem.

The four-feature representation is a major boundary on interpretation. It gives every model the same available inputs and avoids unsupported conversions, but does not characterize performance using richer native features, raw URL encoders or external reputation information. The derived ratio adds no independent information beyond its inputs. Feature grouping reduces within-source representation overlap but cannot certify URL or domain independence. Observed cross-source distribution differences can reflect populations, extraction conventions or both; the current files cannot separate them.

The study also does not establish robustness to severe class imbalance. No controlled SMOTE-versus-weighting or alternative reward-shaping comparison has been completed. Removing these components from the revised primary experiment avoids attributing combined changes to an algorithm, but does not answer their separate effectiveness. Likewise, no direct overestimation measurement, domain-adaptation experiment or probability-calibration study is reported.

Five training seeds on a fixed partition do not represent independent dataset replications. Validation was reused for candidate, checkpoint and threshold selection. The plateau criterion was bounded and applied to one pilot seed, and the latest matched budgets were derived from that diagnostic. The target tables had already informed earlier research decisions; independent confirmation requires a genuinely untouched collection and a protocol fixed before observing its outcomes. No measured difference is presented as statistically significant.

## 7 Conclusion

In a controlled four-feature study of two phishing tables, DQN and DDQN do not show a consistent advantage over supervised references. Random Forest achieves the highest within-source mean AP, while MLP demonstrates the importance of reporting metric-specific rankings. Longer training does not recover a mean Mendeley test-AP advantage for the Q-learning models, and matched discount diagnostics provide no consistent support for future-value bootstrapping. Cross-source false-positive rates remain high and extraction equivalence is unresolved. The findings favour careful baseline comparison, explicit operating-point evaluation and limited claims over interpreting a Q-learning formulation as evidence of adaptation or general robustness.

## Availability of data and materials

The source resources are available through [IEEE DataPort, DOI 10.21227/xngk-3p42](https://doi.org/10.21227/xngk-3p42), and [Mendeley Data, DOI 10.17632/72ptz43s9v.1](https://doi.org/10.17632/72ptz43s9v.1). The evaluated local files and provenance assumptions are identified by hashes in the accompanying audit. Code, aggregate per-seed results, settings and verification records are available in the [research repository](https://github.com/azlan-ismail/phishing-ai-detector-2/tree/research/experiment-redesign). Raw tables, row-level predictions and model checkpoints are not included in that public repository; a file inventory records locally retained artifacts. The [experiment register](experiment-register.md) links each condition to its evidence. The repository is a working research branch rather than an archived release.

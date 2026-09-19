# Dataset provenance and shared-feature audit

## Current decision

The four shared features have support for their intended meanings in the source publications. Exact extraction equivalence is not established. Keep the existing five-seed results as exploratory and do not apply a guessed correction to either dataset. No training, threshold, split or existing result was changed in this audit.

The user instructs us to assume that `ISCX_Phishing.csv` is the IEEE DataPort `ISCXURL2016.csv` renamed without content changes. This is an accepted user-provided assumption, not an independently verified file match. The [IEEE record](https://ieee-dataport.org/documents/iscx-url-2016) identifies DOI 10.21227/xngk-3p42. Original ISCX research is described by [UNB](https://www.unb.ca/cic/datasets/url-2016.html).

The [Mendeley version-1 record](https://data.mendeley.com/datasets/72ptz43s9v/1) explicitly documents **0 = legitimate and 1 = phishing**. The supplied file's 58,645 rows, 111 predictors and class counts (27,998 legitimate; 30,647 phishing) match its small variant. This supports variant identification but does not constitute byte-level identity verification. Earlier reports calling the published Mendeley label convention unresolved are superseded by this source check. Historical run manifests are retained unchanged.

## Feature dictionary and evidence

| Shared feature | ISCX field | Mendeley field | Evidence and remaining limit |
|---|---|---|---|
| URL length | `urlLen` | `length_url` | Both publications describe URL length. Mendeley specifies character count. Scheme inclusion, decoding and normalization equivalence are unverified. |
| Domain length | `domainlength` | `domain_length` | ISCX describes domain length; Mendeley specifies domain character count. Host parsing, subdomains, ports and Unicode handling need extraction-level evidence. |
| URL dot count | `NumberofDotsinURL` | `qty_dot_url` | ISCX names the feature in Table 1; Mendeley defines counts of period characters in the URL. Identical treatment of encoded dots and URL normalization is unverified. |
| Domain/URL ratio | Recompute `domainlength / urlLen` | Recompute `domain_length / length_url` | Identical formula in the current preparation, also described by ISCX. It inherits uncertainty in both input lengths. |

Sources: [Mamun et al., 2016, Section 3.1 and Table 1](https://cyberlab.usask.ca/papers/Mamun2016_Chapter_DetectingMaliciousURLsUsingLex.pdf); [Vrbancic et al., 2020, Tables 1-2](https://www.iztok-jr-fister.eu/static/publications/288.pdf). High-level definitions support candidate mapping, not complete implementation equivalence. The IEEE README link could not be retrieved during this check.

## Actual Python audit

The read-only script checks the supplied whole tables and records input SHA-256 values in [aggregate evidence](../research/audits/feature-compatibility.json). This is post-benchmark descriptive analysis using all rows and labels, not an independent confirmation or source-only model-selection exercise.

Both tables have zero nonfinite values, nonpositive lengths, negative dot counts or fractional counts in the three shared raw fields. No domain length exceeds its URL length; no dot count exceeds URL length. All 15,367 ISCX stored domain/URL ratios match direct recomputation within relative tolerance 1e-6 and absolute tolerance 1e-8. These checks establish numerical consistency only.

| Dataset and class | Rows | Median URL length | Median domain length | Median dot count | Median domain/URL ratio | URL length equals domain length |
|---|---:|---:|---:|---:|---:|---:|
| ISCX benign | 7,781 | 73 | 12 | 1 | 0.1594 | 0 |
| ISCX phishing | 7,586 | 61 | 19 | 3 | 0.3250 | 0 |
| Mendeley benign | 27,998 | 19 | 17 | 2 | 1.0000 | 16,760 |
| Mendeley phishing | 30,647 | 47 | 17 | 2 | 0.3846 | 740 |

The class relationship reverses for URL length and domain/URL ratio. Mendeley has many benign rows with equal URL and domain lengths, while ISCX has none. This is consistent with different URL populations or representation conventions; the numeric tables cannot distinguish those explanations. It makes a source-specific decision rule unreliable under transfer and is a plausible contributor to the observed failures, not proof of their cause.

Secondary diagnostics also show that naive character accounting cannot safely reconstruct a canonical URL: no ISCX row satisfies URL length = stored letter + digit + symbol counts. In Mendeley, only 2,826 of 7,416 rows with nonnegative dot counts for all four components satisfy whole-URL dots = domain + directory + file + parameter dots. Component overlap and counting conventions are not established, so these are diagnostic mismatches, not proof that either dataset is corrupt. Do not repair whole-URL fields by summing these components.

## Consequences for the experiments

1. Keep the published numeric labels: there is no evidence supporting label reversal as a transfer remedy.
2. Retain the current four-feature benchmark as an exploratory comparison with documented extraction uncertainty. Scaling does not resolve incompatible definitions or reversed class relationships.
3. Do not subtract a guessed protocol length or invent reconstructed URL features. Obtain original extractors/raw URLs to verify those conventions before a harmonized primary transfer experiment.
4. A prespecified source-only ablation of URL length and the derived ratio can test model dependence on them. It cannot prove feature compatibility, and choosing the winning feature set from these target results would require new confirmation data.
5. Further within-source tuning can proceed under this scope. Claims about general cross-dataset detection need harmonized extraction and independent confirmation.

Reproduction (use the user's local source paths):

```text
python scripts/audit_feature_compatibility.py --iscx "path/to/ISCX_Phishing.csv" --mendeley "path/to/Mendeley_dataset.csv" --output "work/feature-compatibility.json"
```

The original datasets remain unchanged. Only code, documentation and aggregate diagnostics are suitable for this PR.

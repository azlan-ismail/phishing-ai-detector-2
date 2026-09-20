"""Read-only consistency diagnostics; no feature correction or model selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(path, dataset):
    frame = pd.read_csv(path, low_memory=False)
    cols = ['urlLen', 'domainlength', 'NumberofDotsinURL'] if dataset == 'iscx' else ['length_url', 'domain_length', 'qty_dot_url']
    values = frame[cols].astype(float)
    values.columns = ['url_length', 'domain_length', 'dot_count_url']
    url, domain, dots = (values[c] for c in values)
    labels = frame['URL_Type_obf_Type'] if dataset == 'iscx' else frame['phishing'].map({0: 'benign', 1: 'phishing'})
    result = {'rows': len(frame), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              'class_counts': labels.value_counts().to_dict(),
              'nonfinite_cells': int((~np.isfinite(values)).sum().sum()),
              'nonpositive_url_length': int((url <= 0).sum()),
              'nonpositive_domain_length': int((domain <= 0).sum()),
              'negative_dot_count': int((dots < 0).sum()),
              'fractional_cells': int(((values % 1 != 0) & np.isfinite(values)).sum().sum()),
              'domain_longer_than_url': int((domain > url).sum()),
              'dots_exceed_url_length': int((dots > url).sum()),
              'by_class': {}}
    values['domain_url_ratio'] = domain / url
    for label in ['benign', 'phishing']:
        group = values.loc[labels == label]
        result['by_class'][label] = {'rows': len(group),
            'quantiles': {col: {str(q): float(v) for q, v in group[col].quantile([0, .25, .5, .75, 1]).items()} for col in group},
            'url_equals_domain': int((group.url_length == group.domain_length).sum())}
    if dataset == 'iscx':
        ratio = frame['domainUrlRatio']
        result['stored_ratio_matches_recomputed_rtol_1e-6_atol_1e-8'] = int(np.isclose(ratio, domain / url, rtol=1e-6, atol=1e-8).sum())
        result['url_character_accounting'] = {str(offset): int((url - frame['URL_Letter_Count'] - frame['URL_DigitCount'] - frame['SymbolCount_URL'] == offset).sum()) for offset in [0, 7, 8]}
        result['domain_character_accounting_matches'] = int((domain == frame['host_letter_count'] + frame['host_DigitCount'] + frame['SymbolCount_Domain']).sum())
    else:
        result['domain_dots_exceed_whole_url_dots'] = int((frame.qty_dot_domain > dots).sum())
        result['url_dot_parts_match_where_all_parts_nonnegative'] = {}
        parts = frame[['qty_dot_domain', 'qty_dot_directory', 'qty_dot_file', 'qty_dot_params']]
        valid = (parts >= 0).all(axis=1)
        result['url_dot_parts_match_where_all_parts_nonnegative'] = {'eligible_rows': int(valid.sum()), 'matches': int((parts.loc[valid].sum(axis=1) == dots.loc[valid]).sum())}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--iscx', required=True)
    p.add_argument('--mendeley', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = {'scope': 'Whole-table descriptive audit, not training or hyperparameter selection. Differences do not prove extraction bugs or their cause.',
              'iscx_rename_assumption': 'User states to assume ISCX_Phishing.csv is ISCXURL2016.csv renamed without content changes; not independently byte-verified.',
              'datasets': {name: audit(getattr(a, name), name) for name in ['iscx', 'mendeley']}}
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

"""Compare fixed-partition ablation metrics and verify paired sample identities."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare(baseline, ablation, prepared, output):
    baseline, ablation, prepared, output = map(Path, [baseline, ablation, prepared, output])
    configs = [json.loads((p / 'run.json').read_text()) for p in [baseline, ablation]]
    for key in ['seeds', 'updates', 'trees', 'threads', 'gamma', 'weighting', 'batch_size', 'learning_rate', 'preparation_manifest_sha256', 'budget_selection_sha256', 'hidden_layers', 'input_order', 'fpr_limit', 'software']:
        if configs[0][key] != configs[1][key]:
            raise ValueError('Unmatched comparison setting: ' + key)
    if configs[0].get('feature_condition', 'all') != 'all' or configs[1]['feature_condition'] != 'without_url_length_ratio':
        raise ValueError('Unexpected comparison conditions')
    keys = ['source', 'target', 'model', 'seed', 'operating_point']
    a, b = [pd.read_csv(p / 'metrics.csv', float_precision='round_trip').set_index(keys).sort_index() for p in [baseline, ablation]]
    if not a.index.equals(b.index):
        raise ValueError('Unmatched metric rows')
    checked = 0
    for source, model, seed in a.reset_index()[['source','model','seed']].drop_duplicates().itertuples(index=False, name=None):
        sub = f'{source}_{model}_{seed}'
        for filename in ['validation_predictions.csv', 'iscx_test_predictions.csv', 'mendeley_test_predictions.csv']:
            left, right = [pd.read_csv(p / sub / filename, usecols=['sample_id','label']) for p in [baseline, ablation]]
            if not left.equals(right):
                raise ValueError('Unmatched sample identities/labels: ' + sub + '/' + filename)
            checked += 1
    overlap = {}
    for source in ['iscx', 'mendeley']:
        train, val, test = [pd.read_csv(prepared / f'{source}_to_{source}_{split}.csv') for split in ['train','validation','test']]
        def pairs(frame):
            return list(frame[['domain_length','dot_count_url']].itertuples(index=False, name=None))
        known = set(pairs(train))
        overlap[source] = {split: {'rows': len(frame), 'rows_matching_training_retained_features': sum(x in known for x in pairs(frame))} for split, frame in [('validation',val), ('test',test)]}
    metrics = ['average_precision','roc_auc','f1','recall','precision','fpr','balanced_accuracy','predicted_malicious_fraction']
    delta = b[metrics] - a[metrics]
    output.mkdir(parents=True, exist_ok=True)
    delta.reset_index().to_csv(output / 'paired-feature-differences.csv', index=False)
    summary = delta.groupby(['source','target','model','operating_point']).agg(['mean','std','count'])
    summary.columns = ['_'.join(x) for x in summary.columns]
    summary.reset_index().to_csv(output / 'paired-feature-summary.csv', index=False)
    evidence = {'direction': 'ablation minus baseline', 'matched_prediction_files': checked, 'matched_metric_rows': len(a),
        'retained_feature_collisions': overlap, 'partition_policy': 'Original four-feature grouping retained for pairing; collisions after feature removal are not duplicate URLs or proof of leakage.',
        'inference': 'Descriptive fixed-partition seed comparisons; post-hoc feature hypothesis and no target-based selection or significance claim.'}
    (output / 'comparison-verification.json').write_text(json.dumps(evidence, indent=2) + '\n')
    table = b[['average_precision','f1','fpr']].groupby(['source','target','model','operating_point']).mean()
    print(table.xs('default', level='operating_point').round(4).to_string())
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline', 'ablation', 'prepared', 'output']:
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    compare(args.baseline, args.ablation, args.prepared, args.output)

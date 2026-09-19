"""Recompute saved benchmark metrics and source-validation threshold choices.

Writes aggregate evidence only. Original benchmark predictions are read-only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_research_benchmark import evaluate, select_thresholds, save_json, file_hash, MODELS


def verify(run, output):
    run, output = Path(run), Path(output)
    config = json.loads((run / 'run.json').read_text())
    completion = json.loads((run / 'completion.json').read_text())
    metrics = pd.read_csv(run / 'metrics.csv', float_precision='round_trip')
    keys = ['source', 'target', 'model', 'seed', 'operating_point']
    expected = 2 * 2 * len(MODELS) * len(config['seeds']) * 3
    if not completion['complete'] or len(metrics) != expected or metrics.duplicated(keys).any():
        raise ValueError('Incomplete or duplicate metric rows')
    if set(metrics['seed']) != set(config['seeds']):
        raise ValueError('Unexpected seeds')
    checked = 0
    for (source, model, seed), part in metrics.groupby(['source', 'model', 'seed']):
        folder = run / f'{source}_{model}_{seed}'
        selection = json.loads((folder / 'selection.json').read_text())
        if selection['threshold_source'] != 'source_validation_only':
            raise ValueError('Unexpected threshold-selection source')
        val = pd.read_csv(folder / 'validation_predictions.csv', float_precision='round_trip')
        thresholds = select_thresholds(val['label'].to_numpy(), val['score'].to_numpy(),
                                       selection['thresholds']['default'], config['fpr_limit'])
        if thresholds != selection['thresholds']:
            raise ValueError('Saved thresholds do not match validation-only selection')
        for target, rows in part.groupby('target'):
            predictions = pd.read_csv(folder / f'{target}_test_predictions.csv', float_precision='round_trip')
            if set(val.sample_id) & set(predictions.sample_id):
                raise ValueError('Validation/test sample overlap')
            for _, row in rows.iterrows():
                op = row['operating_point']
                threshold = thresholds[op]
                if row['threshold'] != threshold:
                    raise ValueError('Test threshold differs from frozen validation selection')
                actual = evaluate(predictions['label'].to_numpy(), predictions['score'].to_numpy(), threshold)
                for key, value in actual.items():
                    if not np.isclose(row[key], value, rtol=1e-10, atol=1e-12):
                        raise ValueError('Metric mismatch: ' + key)
                if not np.array_equal(predictions['prediction_' + op], (predictions.score >= threshold).astype(int)):
                    raise ValueError('Prediction mismatch')
                checked += 1
    budgets = []
    for source in ['iscx', 'mendeley']:
        for seed in config['seeds']:
            left = json.loads((run / f'{source}_dqn_{seed}/selection.json').read_text())['training']
            right = json.loads((run / f'{source}_ddqn_{seed}/selection.json').read_text())['training']
            if left != right:
                raise ValueError('DQN/DDQN training budget mismatch')
            budgets.append({'source': source, 'seed': seed, **left})
    output.mkdir(parents=True, exist_ok=True)
    numeric = ['average_precision', 'roc_auc', 'f1', 'precision', 'recall', 'fpr', 'balanced_accuracy']
    summary = metrics.groupby(['source', 'target', 'model', 'operating_point'])[numeric].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(output / 'summary.csv', index=False)
    pair = metrics[metrics.model.isin(['dqn', 'ddqn'])].pivot(index=['source','target','seed','operating_point'], columns='model', values=numeric)
    differences = pd.DataFrame({metric + '_ddqn_minus_dqn': pair[(metric, 'ddqn')] - pair[(metric, 'dqn')] for metric in numeric})
    differences.reset_index().to_csv(output / 'paired-differences.csv', index=False)
    paired_summary = differences.groupby(['source','target','operating_point']).agg(['mean','std','count'])
    paired_summary.columns = ['_'.join(c) for c in paired_summary.columns]
    paired_summary.reset_index().to_csv(output / 'paired-summary.csv', index=False)
    result = {'purpose': config['purpose'], 'seeds': config['seeds'], 'verified_metric_rows': checked,
              'fit_configurations': 2 * len(MODELS) * len(config['seeds']),
              'source_target_evaluations': 4 * len(MODELS) * len(config['seeds']),
              'validation_thresholds_reproduced': True, 'predictions_recompute_all_metrics': True,
              'matched_rl_budgets': budgets, 'configuration': config,
              'metric_file_sha256': file_hash(run / 'metrics.csv'),
              'uncertainty_scope': 'Sample SD across training seeds on fixed partitions; not dataset or test-sampling uncertainty. No significance claim.',
              'verifier_sha256': file_hash(__file__)}
    save_json(output / 'verification.json', result)
    print(json.dumps({k: result[k] for k in ['seeds','verified_metric_rows','fit_configurations','source_target_evaluations']}, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    verify(args.run, args.output)


if __name__ == '__main__':
    main()

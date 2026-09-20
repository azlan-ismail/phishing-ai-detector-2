"""Verify a fixed-budget gamma-zero diagnostic against gamma 0.99."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np


def compare(baseline, ablation, output):
    baseline, ablation, output = map(Path, [baseline, ablation, output])
    ca, cb = [json.loads((p / 'run.json').read_text()) for p in [baseline, ablation]]
    for k in ['selected_settings', 'tuning_selection_sha256']:
        if ca.get(k) != cb.get(k):
            raise ValueError('Unmatched inherited selection: ' + k)
    for k in ['seeds', 'updates', 'trees', 'threads', 'weighting', 'batch_size', 'learning_rate', 'preparation_manifest_sha256', 'hidden_layers', 'input_order', 'fpr_limit', 'software', 'replay_capacity', 'target_interval_updates', 'gradient_clip_norm', 'epsilon_initial', 'epsilon_min', 'epsilon_decay_per_update']:
        if ca[k] != cb[k]:
            raise ValueError('Unmatched setting: ' + k)
    if ca['gamma'] != .99 or cb['gamma'] != 0 or any(c.get('feature_condition', 'all') != 'all' for c in [ca, cb]):
        raise ValueError('Unexpected discount or feature condition')
    keys = ['source','target','model','seed','operating_point']
    a, b = [pd.read_csv(p / 'metrics.csv', float_precision='round_trip').set_index(keys).sort_index() for p in [baseline, ablation]]
    if not a.index.equals(b.index) or a.index.has_duplicates or len(a) != 360:
        raise ValueError('Incomplete or unpaired metrics')
    identity_checks = control_checks = equality_checks = 0
    control_max_score_difference = 0.0
    control_changed_decisions = 0
    for source in ['iscx','mendeley']:
        for seed in ca['seeds']:
            for model in ['always_malicious','logistic','random_forest','mlp','dqn','ddqn']:
                if model in ['mlp','dqn','ddqn']:
                    actual = [json.loads((p / f'{source}_{model}_{seed}' / 'selection.json').read_text())['training'] for p in [baseline, ablation]]
                    if actual[0] != actual[1]:
                        raise ValueError('Unmatched actual training exposure: ' + model)
                for filename in ['validation_predictions.csv','iscx_test_predictions.csv','mendeley_test_predictions.csv']:
                    left, right = [pd.read_csv(p / f'{source}_{model}_{seed}' / filename, float_precision='round_trip') for p in [baseline, ablation]]
                    if not left[['sample_id','label']].equals(right[['sample_id','label']]):
                        raise ValueError('Sample pairing failed')
                    identity_checks += 1
                    if model not in ['dqn','ddqn']:
                        if not np.allclose(left.score, right.score, rtol=0, atol=1e-12):
                            raise ValueError('Non-RL control changed: ' + model)
                        control_max_score_difference = max(control_max_score_difference, float((left.score-right.score).abs().max()))
                        decisions = [c for c in left if c.startswith('prediction_')]
                        control_changed_decisions += int((left[decisions] != right[decisions]).sum().sum())
                        control_checks += 1
            for filename in ['validation_predictions.csv','iscx_test_predictions.csv','mendeley_test_predictions.csv','training.csv']:
                dqn, ddqn = [pd.read_csv(ablation / f'{source}_{model}_{seed}' / filename, float_precision='round_trip') for model in ['dqn','ddqn']]
                if not dqn.equals(ddqn):
                    raise ValueError('Gamma-zero DQN/DDQN mismatch: ' + filename)
                equality_checks += 1
    metrics = ['average_precision','roc_auc','f1','precision','recall','fpr','balanced_accuracy','predicted_malicious_fraction']
    delta = b[metrics] - a[metrics]
    output.mkdir(parents=True, exist_ok=True)
    delta.reset_index().to_csv(output / 'paired-gamma-differences.csv', index=False)
    summary = delta.groupby(['source','target','model','operating_point']).agg(['mean','std','count'])
    summary.columns = ['_'.join(x) for x in summary.columns]
    summary.reset_index().to_csv(output / 'paired-gamma-summary.csv', index=False)
    result = {'direction': 'gamma zero minus gamma 0.99', 'matched_prediction_files': identity_checks,
        'non_rl_score_control_files_within_absolute_1e-12': control_checks,
        'non_rl_max_absolute_score_difference': control_max_score_difference,
        'non_rl_changed_operating_point_decisions': control_changed_decisions,
        'non_rl_max_absolute_metric_differences': {model: {m: float(delta.xs(model, level='model')[m].abs().max()) for m in metrics} for model in ['always_malicious','logistic','random_forest','mlp']},
        'identical_gamma_zero_dqn_ddqn_files': equality_checks,
        'budget': 'Training settings inherited unchanged from gamma-0.99 source-validation selection; gamma-zero not independently tuned.',
        'updates': ca['updates'],
        'selected_settings': ca.get('selected_settings'),
        'actual_neural_training_exposures_matched': True,
        'baseline_run_sha256': hashlib.sha256((baseline / 'run.json').read_bytes()).hexdigest(),
        'ablation_run_sha256': hashlib.sha256((ablation / 'run.json').read_bytes()).hexdigest(),
        'comparison_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Exploratory paired seeds on fixed partitions, not significance or dataset uncertainty.'}
    (output / 'comparison-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','ablation','output']:
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    compare(a.baseline, a.ablation, a.output)

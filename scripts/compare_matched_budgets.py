"""Verify the longer matched-budget run against the original tuned benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_research_benchmark import file_hash, save_json


def compare(baseline, extended, policy, output):
    baseline,extended,policy,output=map(Path,[baseline,extended,policy,output])
    configs=[json.loads((p/'run.json').read_text()) for p in [baseline,extended]]
    a,b=configs
    frozen=json.loads(policy.read_text())
    if b['matched_budget_policy_sha256']!=file_hash(policy):
        raise ValueError('Frozen policy hash mismatch')
    for key in ['seeds','gamma','batch_size','trees','threads','weighting','feature_condition',
                'preparation_manifest_sha256','tuning_selection_sha256','software','input_order','hidden_layers',
                'fpr_limit','replay_capacity','target_interval_updates','gradient_clip_norm',
                'epsilon_initial','epsilon_min','epsilon_decay_per_update']:
        if a[key]!=b[key]:
            raise ValueError('Unexpected setting change: '+key)
    identities=controls=0
    max_difference=0.
    for source in ['iscx','mendeley']:
        for model in ['logistic','random_forest','mlp','dqn','ddqn']:
            old,new=[c['selected_settings'][source][model].copy() for c in configs]
            if model in ['mlp','dqn','ddqn']:
                if old.pop('updates')!=8000 or new.pop('updates')!=frozen['source_updates'][source]:
                    raise ValueError('Unexpected neural update count')
            if old!=new:
                raise ValueError('Changed non-budget model setting')
        for seed in a['seeds']:
            for model in ['always_malicious','logistic','random_forest','mlp','dqn','ddqn']:
                sub=f'{source}_{model}_{seed}'
                if model in ['mlp','dqn','ddqn']:
                    info=json.loads((extended/sub/'selection.json').read_text())['training']
                    history=pd.read_csv(extended/sub/'training.csv')
                    budget=frozen['source_updates'][source]
                    if info['optimizer_updates']!=budget or history['update'].tolist()!=list(range(1,budget+1)):
                        raise ValueError('Actual update/history mismatch')
                for filename in ['validation_predictions.csv','iscx_test_predictions.csv','mendeley_test_predictions.csv']:
                    x,y=[pd.read_csv(p/sub/filename,float_precision='round_trip') for p in [baseline,extended]]
                    if not x[['sample_id','label']].equals(y[['sample_id','label']]):
                        raise ValueError('Sample pairing mismatch')
                    identities+=1
                    if model in ['always_malicious','logistic','random_forest']:
                        difference=float((x.score-y.score).abs().max())
                        if difference>1e-12:
                            raise ValueError('Non-neural control scores changed')
                        predictions=[c for c in x if c.startswith('prediction_')]
                        if not x[predictions].equals(y[predictions]):
                            raise ValueError('Non-neural control decisions changed')
                        max_difference=max(max_difference,difference);controls+=1
            dqn,ddqn=[json.loads((extended/f'{source}_{m}_{seed}'/'selection.json').read_text())['training'] for m in ['dqn','ddqn']]
            if dqn!=ddqn:
                raise ValueError('RL training exposures not matched')
    keys=['source','target','model','seed','operating_point']
    x,y=[pd.read_csv(p/'metrics.csv',float_precision='round_trip').set_index(keys).sort_index() for p in [baseline,extended]]
    if not x.index.equals(y.index) or x.index.has_duplicates or len(x)!=360:
        raise ValueError('Unpaired metrics')
    metrics=['average_precision','roc_auc','f1','precision','recall','fpr','balanced_accuracy']
    delta=y[metrics]-x[metrics]
    for model in ['always_malicious','logistic','random_forest']:
        if (delta.xs(model,level='model').abs().to_numpy()>1e-12).any():
            raise ValueError('Control metrics changed')
    output.mkdir(parents=True,exist_ok=True)
    delta.reset_index().to_csv(output/'paired-budget-differences.csv',index=False)
    summary=delta.groupby(['source','target','model','operating_point']).agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(output/'paired-budget-summary.csv',index=False)
    result={'direction':'longer matched budgets minus tuned 8000-update reference',
        'paired_prediction_files':identities,'non_neural_control_files':controls,
        'control_max_score_difference':max_difference,'control_decisions_and_metrics_unchanged':True,
        'actual_neural_budgets_and_histories_verified':True,'rl_training_exposures_matched':True,
        'source_updates':frozen['source_updates'],'policy_sha256':file_hash(policy),
        'baseline_config_sha256':file_hash(baseline/'run.json'),'extended_config_sha256':file_hash(extended/'run.json'),
        'comparison_script_sha256':file_hash(__file__),
        'scope':'Exploratory paired training seeds on fixed partitions; equal updates are not equal compute.'}
    save_json(output/'comparison-verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','extended','policy','output']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args();compare(a.baseline,a.extended,a.policy,a.output)

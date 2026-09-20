"""Recompute trial objectives and selections, then compare paired tuned tests."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tune_source_models import choose
from run_research_benchmark import file_hash, save_json


def verify(search, baseline, tuned, output):
    search, baseline, tuned, output = map(Path, [search,baseline,tuned,output])
    selection = json.loads((search/'selection.json').read_text())
    rows = json.loads((search/'trials.json').read_text())
    if len(rows) != 60:
        raise ValueError('Incomplete search')
    for row in rows:
        pred = pd.read_csv(search/f"{row['source']}_{row['model']}_{row['candidate']}"/'validation_scores.csv',float_precision='round_trip')
        if not np.isclose(average_precision_score(pred.label,pred.score),row['validation_ap'],rtol=0,atol=1e-12):
            raise ValueError('Trial AP mismatch')
    learning = []
    for source in ['iscx','mendeley']:
        for model in ['logistic','random_forest','mlp','dqn','ddqn']:
            part = [r for r in rows if r['source']==source and r['model']==model]
            if len(part)!=6 or len({r['candidate'] for r in part})!=6:
                raise ValueError('Candidate coverage mismatch')
            if choose(part)['settings'] != selection['selected'][source][model]:
                raise ValueError('Selected setting not validation winner')
            if model in ['mlp','dqn','ddqn']:
                for rate in selection['neural_learning_rates']:
                    curve = sorted([r for r in part if r['settings']['learning_rate']==rate],key=lambda r:r['settings']['updates'])
                    learning.append({'source':source,'model':model,'learning_rate':rate,
                        'ap_1000':curve[0]['validation_ap'],'ap_4000':curve[1]['validation_ap'],'ap_8000':curve[2]['validation_ap'],
                        'last_interval_ap_change':curve[2]['validation_ap']-curve[1]['validation_ap'],
                        'convergence_established':False})
    ca, cb = [json.loads((p/'run.json').read_text()) for p in [baseline,tuned]]
    if cb['tuning_selection_sha256'] != file_hash(search/'selection.json') or cb['selected_settings'] != selection['selected']:
        raise ValueError('Run does not match selected settings')
    for k in ['preparation_manifest_sha256','seeds','gamma','weighting','batch_size','trees','input_order','hidden_layers','software','fpr_limit']:
        if ca[k] != cb[k]:
            raise ValueError('Pairing configuration mismatch: '+k)
    checked = 0
    for source in ['iscx','mendeley']:
        for seed in ca['seeds']:
            for model in ['always_malicious','logistic','random_forest','mlp','dqn','ddqn']:
                sub = f'{source}_{model}_{seed}'
                for f in ['validation_predictions.csv','iscx_test_predictions.csv','mendeley_test_predictions.csv']:
                    a,b = [pd.read_csv(p/sub/f,usecols=['sample_id','label']) for p in [baseline,tuned]]
                    if not a.equals(b):
                        raise ValueError('Sample pairing mismatch')
                    checked += 1
                if model in ['mlp','dqn','ddqn']:
                    actual = json.loads((tuned/sub/'selection.json').read_text())['training']['optimizer_updates']
                    if actual != selection['selected'][source][model]['updates']:
                        raise ValueError('Neural budget mismatch')
    keys=['source','target','model','seed','operating_point']
    a,b=[pd.read_csv(p/'metrics.csv').set_index(keys).sort_index() for p in [baseline,tuned]]
    if not a.index.equals(b.index) or a.index.has_duplicates or len(a)!=360:
        raise ValueError('Metric pairing mismatch')
    metrics=['average_precision','roc_auc','f1','precision','recall','fpr','balanced_accuracy']
    delta=b[metrics]-a[metrics]
    output.mkdir(parents=True,exist_ok=True)
    delta.reset_index().to_csv(output/'paired-tuning-differences.csv',index=False)
    summary=delta.groupby(['source','target','model','operating_point']).agg(['mean','std','count'])
    summary.columns=['_'.join(x) for x in summary.columns]
    summary.reset_index().to_csv(output/'paired-tuning-summary.csv',index=False)
    pd.DataFrame(learning).to_csv(output/'validation-learning-curves.csv',index=False)
    result={'verified_trial_objectives':60,'selected_settings_reproduced':True,'matched_prediction_files':checked,
        'all_neural_selected_budgets_checked':True,'direction':'tuned minus original baseline',
        'inference':'Bounded one-pilot-seed selection, fixed-partition five-seed reporting; no significance or convergence claim.'}
    save_json(output/'comparison-verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['search','baseline','tuned','output']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    verify(a.search,a.baseline,a.tuned,a.output)

"""Six-candidate bounded tuning; opens only source train/validation exports."""
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from run_research_benchmark import load_export, train_neural, score_model, save_json, file_hash


def choose(rows):
    # Fixed exact-score tie rule: earliest enumerated candidate.
    return max(rows, key=lambda r: (r['validation_ap'], -r['candidate']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    prepared, output = Path(a.prepared), Path(a.output)
    if output.exists():
        raise ValueError('Use a new output directory')
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    budgets = [1000,4000,8000]
    rates = [.0003,.001]
    policy = {'preparation_manifest_sha256': file_hash(prepared/'manifest.json'), 'pilot_seed': 101,
              'gamma': .99, 'batch_size': 64, 'weighting': 'none', 'feature_condition': 'all', 'trees': 200,
              'threads': 2, 'objective': 'highest source-validation average precision; exact ties select earliest candidate',
              'neural_budgets': budgets, 'neural_learning_rates': rates,
              'logistic_C': [.001,.01,.1,1.,10.,100.],
              'rf_max_depth': [8,16,None], 'rf_min_samples_leaf': [1,5],
              'candidates_per_model_per_source': 6, 'test_exports_opened': False,
              'search_seed_limitation': 'One pilot seed; final five seeds are distinct. Equal candidate counts, not equal compute.',
              'selector_sha256': file_hash(__file__), 'trainer_sha256': file_hash(Path(__file__).with_name('run_research_benchmark.py'))}
    save_json(output/'policy.json', policy)
    manifest = json.loads((prepared/'manifest.json').read_text())
    records, selected, boundary = [], {}, {}
    for source in ['iscx','mendeley']:
        ids,x,y = load_export(prepared,manifest,source,source,'train')
        vids,vx,vy = load_export(prepared,manifest,source,source,'validation')
        if set(ids) & set(vids):
            raise ValueError('Train/validation overlap')
        selected[source], boundary[source] = {}, {}
        for name in ['logistic','random_forest','mlp','dqn','ddqn']:
            rows = []
            def record(model, settings, candidate, seconds, captured_warnings=None):
                score = score_model(model,name,vx).astype(float)
                folder = output/f'{source}_{name}_{candidate}'
                folder.mkdir()
                pd.DataFrame({'sample_id':vids,'label':vy,'score':score}).to_csv(folder/'validation_scores.csv',index=False)
                if name in ['mlp','dqn','ddqn']:
                    torch.save(model.state_dict(),folder/'weights.pt')
                else:
                    joblib.dump(model,folder/'model.joblib')
                row = {'source':source,'model':name,'candidate':candidate,'settings':settings,
                       'validation_ap':float(average_precision_score(vy,score)), 'seconds':seconds,
                       'warnings':captured_warnings or []}
                save_json(folder/'trial.json',row)
                rows.append(row); records.append(row)
                save_json(output/'trials.json',records)
                print(source,name,settings,'AP',round(row['validation_ap'],6),flush=True)
            if name == 'logistic':
                for candidate,C in enumerate(policy['logistic_C']):
                    start = time.perf_counter()
                    model = LogisticRegression(C=C,max_iter=2000,random_state=101)
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter('always'); model.fit(x,y)
                    record(model,{'C':C},candidate,time.perf_counter()-start,[str(w.message) for w in captured])
            elif name == 'random_forest':
                candidate = 0
                for depth in policy['rf_max_depth']:
                    for leaf in policy['rf_min_samples_leaf']:
                        start = time.perf_counter()
                        model = RandomForestClassifier(n_estimators=200,max_depth=depth,min_samples_leaf=leaf,random_state=101,n_jobs=2)
                        model.fit(x,y)
                        record(model,{'max_depth':depth,'min_samples_leaf':leaf},candidate,time.perf_counter()-start)
                        candidate += 1
            else:
                for ri,rate in enumerate(rates):
                    start = time.perf_counter()
                    def observer(update, model):
                        if update in budgets:
                            record(model,{'learning_rate':rate,'updates':update},ri*3+budgets.index(update),time.perf_counter()-start)
                    _,history,_ = train_neural(x,y,name,101,max(budgets),learning_rate=rate,observer=observer)
                    pd.DataFrame(history).to_csv(output/f'{source}_{name}_lr{rate}_training.csv',index=False)
            best = choose(rows)
            selected[source][name] = best['settings']
            boundary[source][name] = {'selected_candidate':best['candidate'], 'validation_ap':best['validation_ap'],
                'selected_at_largest_neural_budget': best['settings'].get('updates') == max(budgets),
                'convergence_established':False}
            save_json(output/'selection-in-progress.json',{'selected':selected,'diagnostics':boundary})
    save_json(output/'selection.json',{**policy,'selected':selected,'diagnostics':boundary,'completed_candidate_evaluations':len(records)})
    print(json.dumps(selected,indent=2),flush=True)


if __name__ == '__main__':
    main()

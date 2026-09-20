"""Source-validation-only bounded plateau diagnostic; never opens test exports."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import average_precision_score
from run_research_benchmark import load_export, train_neural, score_model, save_json, file_hash


class Plateau:
    def __init__(self, minimum_gain=.001, patience=2):
        self.minimum_gain, self.patience = minimum_gain, patience
        self.anchor, self.stale = None, 0

    def observe(self, score):
        if self.anchor is None or score-self.anchor >= self.minimum_gain:
            self.anchor, self.stale = score, 0
        else:
            self.stale += 1
        return self.stale >= self.patience


def run(prepared, reference, output):
    prepared, reference, output = map(Path,[prepared,reference,output])
    if output.exists():
        raise ValueError('Use a new output directory')
    manifest=json.loads((prepared/'manifest.json').read_text())
    prior=json.loads((reference/'selection.json').read_text())
    if prior['preparation_manifest_sha256']!=file_hash(prepared/'manifest.json'):
        raise ValueError('Reference preparation mismatch')
    for source in ['iscx','mendeley']:
        for name in ['mlp','dqn','ddqn']:
            if prior['selected'][source][name]!={'learning_rate':.001,'updates':8000}:
                raise ValueError('Unexpected reference settings')
    if prior['gamma']!=.99 or prior['pilot_seed']!=101:
        raise ValueError('Unexpected reference gamma or seed')
    output.mkdir(parents=True)
    policy={'purpose':'exploratory source-validation plateau diagnostic',
        'seed':101,'seed_scope':'Original pilot seed reused; not five-seed convergence evidence.',
        'models':['mlp','dqn','ddqn'],'sources':['iscx','mendeley'],
        'checkpoints':list(range(8000,32001,4000)),'minimum_gain':.001,'patience':2,
        'stopping_rule':'At each checkpoint, reset patience if AP exceeds the last meaningful best by at least .001; stop after two failures, or at 32000.',
        'selection_rule':'Highest observed validation AP, exact ties earliest checkpoint.',
        'gamma':.99,'learning_rate':.001,'batch_size':64,'weighting':'none','threads':2,
        'preparation_manifest_sha256':file_hash(prepared/'manifest.json'),
        'reference_selection_sha256':file_hash(reference/'selection.json'),
        'script_sha256':file_hash(__file__),
        'trainer_sha256':file_hash(Path(__file__).with_name('run_research_benchmark.py')),
        'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                    'sklearn':sklearn.__version__,'torch':torch.__version__},
        'test_exports_opened':False,'convergence_claim':'Operational validation plateau only; no mathematical or multi-seed convergence claim.'}
    save_json(output/'policy.json',policy)
    torch.set_num_threads(2)
    opened,results=[],[]
    for source in policy['sources']:
        arrays=[]
        for split in ['train','validation']:
            filename=f'{source}_to_{source}_{split}.csv'
            ids,x,y=load_export(prepared,manifest,source,source,split)
            opened.append({'file':filename,'sha256':file_hash(prepared/filename),'rows':len(y)})
            arrays.append((ids,x,y))
        save_json(output/'opened-exports.json',opened)
        (ids,x,y),(vids,vx,vy)=arrays
        if set(ids)&set(vids):
            raise ValueError('Training/validation overlap')
        for name in policy['models']:
            folder=output/f'{source}_{name}'
            folder.mkdir()
            plateau=Plateau(policy['minimum_gain'],policy['patience'])
            rows=[]
            start=time.perf_counter()
            def observer(update, model):
                if update not in policy['checkpoints']:
                    return False
                scores=score_model(model,name,vx).astype(float)
                score=float(average_precision_score(vy,scores))
                checkpoint=folder/str(update)
                checkpoint.mkdir()
                pd.DataFrame({'sample_id':vids,'label':vy,'score':scores}).to_csv(checkpoint/'validation_scores.csv',index=False)
                torch.save(model.state_dict(),checkpoint/'weights.pt')
                if update==8000:
                    old=pd.read_csv(reference/f'{source}_{name}_5'/'validation_scores.csv',float_precision='round_trip')
                    np.testing.assert_array_equal(old.sample_id.to_numpy(),vids)
                    np.testing.assert_array_equal(old.label.to_numpy(),vy)
                    np.testing.assert_array_equal(old.score.to_numpy(),scores)
                stop=plateau.observe(score)
                rows.append({'updates':update,'validation_ap':score,'meaningful_best_ap':plateau.anchor,
                             'consecutive_small_gains':plateau.stale,'plateau_rule_met':stop,
                             'cumulative_seconds':time.perf_counter()-start})
                save_json(folder/'checkpoints.json',rows)
                print(source,name,update,'AP',round(score,6),'plateau',stop,flush=True)
                return stop
            _,history,training=train_neural(x,y,name,101,32000,gamma=.99,learning_rate=.001,
                                            batch_size=64,weighting='none',observer=observer)
            pd.DataFrame(history).to_csv(folder/'training.csv',index=False)
            best=max(rows,key=lambda row:(row['validation_ap'],-row['updates']))
            result={'source':source,'model':name,'selected_updates':best['updates'],
                    'selected_validation_ap':best['validation_ap'],'initial_8000_ap':rows[0]['validation_ap'],
                    'gain_over_8000':best['validation_ap']-rows[0]['validation_ap'],
                    'stop_updates':training['optimizer_updates'],
                    'stop_reason':'validation_plateau_rule' if rows[-1]['plateau_rule_met'] else 'maximum_budget',
                    'prefix_8000_exactly_reproduced':True,'training':training,
                    'checkpoints':rows,'convergence_established':False}
            save_json(folder/'result.json',result)
            results.append(result)
            save_json(output/'results.json',results)
    save_json(output/'completion.json',{'complete':True,'trajectories':len(results),
        'checkpoint_evaluations':sum(len(r['checkpoints']) for r in results),'test_evaluations':0})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','reference','output']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    run(a.prepared,a.reference,a.output)

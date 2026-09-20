"""Recompute validation objectives, checkpoint selection and the plateau rule."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from run_research_benchmark import file_hash, save_json


def verify(run, output):
    run,output=map(Path,[run,output])
    policy=json.loads((run/'policy.json').read_text())
    completion=json.loads((run/'completion.json').read_text())
    results=json.loads((run/'results.json').read_text())
    if not completion['complete'] or len(results)!=6:
        raise ValueError('Incomplete trajectories')
    opened=json.loads((run/'opened-exports.json').read_text())
    expected={f'{s}_to_{s}_{p}.csv' for s in ['iscx','mendeley'] for p in ['train','validation']}
    if {r['file'] for r in opened}!=expected or len(opened)!=4:
        raise ValueError('Unexpected prepared export access record')
    if {(r['source'],r['model']) for r in results}!={(s,m) for s in policy['sources'] for m in policy['models']}:
        raise ValueError('Unexpected model/source coverage')
    summaries,curves=[],[]
    for result in results:
        folder=run/f"{result['source']}_{result['model']}"
        rows=json.loads((folder/'checkpoints.json').read_text())
        if rows!=result['checkpoints']:
            raise ValueError('Checkpoint record mismatch')
        anchor=None
        stale=0
        identity=None
        for i,row in enumerate(rows):
            if row['updates']!=policy['checkpoints'][i]:
                raise ValueError('Skipped checkpoint')
            frame=pd.read_csv(folder/str(row['updates'])/'validation_scores.csv',float_precision='round_trip')
            if identity is not None and not identity.equals(frame[['sample_id','label']]):
                raise ValueError('Validation identity changed')
            identity=frame[['sample_id','label']]
            ap=float(average_precision_score(frame.label,frame.score))
            if not np.isclose(ap,row['validation_ap'],rtol=0,atol=1e-12):
                raise ValueError('AP mismatch')
            if anchor is None or ap-anchor>=policy['minimum_gain']:
                anchor,stale=ap,0
            else:
                stale+=1
            stop=stale>=policy['patience']
            if stop!=row['plateau_rule_met'] or stale!=row['consecutive_small_gains'] or anchor!=row['meaningful_best_ap']:
                raise ValueError('Plateau rule mismatch')
            if stop and i!=len(rows)-1:
                raise ValueError('Training continued after stop rule')
            curves.append({'source':result['source'],'model':result['model'],**row})
        if not rows[-1]['plateau_rule_met'] and rows[-1]['updates']!=max(policy['checkpoints']):
            raise ValueError('Premature stop')
        selected=max(rows,key=lambda r:(r['validation_ap'],-r['updates']))
        if selected['updates']!=result['selected_updates'] or selected['validation_ap']!=result['selected_validation_ap']:
            raise ValueError('Selected checkpoint mismatch')
        history=pd.read_csv(folder/'training.csv')
        if history['update'].tolist()!=list(range(1,result['stop_updates']+1)):
            raise ValueError('History or actual update count mismatch')
        if len(history)!=rows[-1]['updates'] or len(history)!=result['training']['optimizer_updates']:
            raise ValueError('Training budget mismatch')
        if int(history.training_sample_exposures.iloc[-1])!=result['training']['sample_exposures']:
            raise ValueError('Exposure mismatch')
        summaries.append({k:v for k,v in result.items() if k not in ['checkpoints','training']})
    if len(curves)!=completion['checkpoint_evaluations']:
        raise ValueError('Checkpoint total mismatch')
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
    pd.DataFrame(curves).to_csv(output/'validation-learning-curves.csv',index=False)
    evidence={'verified_trajectories':6,'verified_checkpoints':len(curves),
        'prepared_export_access_record':'Only four source train/validation exports',
        'test_evaluations':completion['test_evaluations'],
        'actual_stopping_and_selection_reproduced':True,
        'policy_sha256':file_hash(run/'policy.json'),'results_sha256':file_hash(run/'results.json'),
        'verifier_sha256':file_hash(__file__),
        'scope':'Single reused pilot seed; no test performance, significance or mathematical convergence claim.'}
    save_json(output/'verification.json',evidence)
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); verify(a.run,a.output)

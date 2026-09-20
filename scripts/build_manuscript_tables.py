"""Insert manuscript tables derived directly from the verified latest run."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd


def build(manuscript, metrics, summary, evidence):
    manuscript,metrics,summary,evidence=map(Path,[manuscript,metrics,summary,evidence])
    raw=pd.read_csv(metrics,float_precision='round_trip')
    saved=pd.read_csv(summary,float_precision='round_trip')
    grouped=raw.groupby(['source','target','model','operating_point'])
    checked=0
    for _,r in saved.iterrows():
        g=grouped.get_group((r.source,r.target,r.model,r.operating_point))
        for m in ['average_precision','f1','precision','recall','fpr']:
            if abs(g[m].mean()-r[m+'_mean'])>1e-12 or abs(g[m].std(ddof=1)-r[m+'_std'])>1e-12:
                raise ValueError('Saved summary mismatch')
            checked+=1
    def value(source,target,model,op,metric):
        g=grouped.get_group((source,target,model,op))[metric]
        return f'{g.mean():.4f} ± {g.std(ddof=1):.4f}'
    within=['**Table 1. Within-source results, mean ± sample SD across five seeds. F1 uses the default rule.**\n',
        '| Model | ISCX AP | ISCX F1 | Mendeley AP | Mendeley F1 |',
        '|---|---:|---:|---:|---:|']
    for m,label in [('always_malicious','Always phishing'),('logistic','Logistic regression'),('random_forest','Random Forest'),('mlp','MLP'),('dqn','DQN'),('ddqn','DDQN')]:
        within.append('| '+' | '.join([label]+[value(s,s,m,'default',v) for s in ['iscx','mendeley'] for v in ['average_precision','f1']])+' |')
    transfer=['**Table 2. DDQN transfer results, mean ± sample SD across five seeds. The last three columns use the source-validation 1% FPR constraint.**\n',
        '| Source → target | AP | Default F1 | Precision | Recall | FPR |',
        '|---|---:|---:|---:|---:|---:|']
    for source,target in [('iscx','mendeley'),('mendeley','iscx')]:
        cells=[value(source,target,'ddqn','default',v) for v in ['average_precision','f1']]
        cells += [value(source,target,'ddqn','validation_fpr_limit',v) for v in ['precision','recall','fpr']]
        transfer.append('| '+' | '.join([source.upper()+' → '+target.upper()]+cells)+' |')
    text=manuscript.read_text(encoding='utf-8')
    for marker,lines in [('{{WITHIN_TABLE}}',within),('{{TRANSFER_TABLE}}',transfer)]:
        if marker in text:
            text=text.replace(marker,'\n'.join(lines))
        else:
            pattern=re.escape(lines[0].strip())+r'\n\n(?:\|[^\n]*(?:\n|$))+'
            text,count=re.subn(pattern,lambda match:'\n'.join(lines)+'\n',text)
            if count!=1:
                raise ValueError('Missing or ambiguous table location')
    manuscript.write_text(text,encoding='utf-8')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    evidence.parent.mkdir(parents=True,exist_ok=True)
    report={'source_metrics_sha256':sha(metrics),'verified_summary_sha256':sha(summary),
        'manuscript_sha256':sha(manuscript),'script_sha256':sha(Path(__file__)),
        'metric_summary_pairs_checked':checked,'within_table_rows':6,'transfer_table_rows':2,
        'scope':'Tables recomputed from saved per-seed metrics; manuscript interpretation requires author review.'}
    evidence.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manuscript','metrics','summary','evidence']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args();build(a.manuscript,a.metrics,a.summary,a.evidence)

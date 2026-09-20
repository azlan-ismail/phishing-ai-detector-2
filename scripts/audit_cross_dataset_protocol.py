"""Read-only reconstruction of source-fitted cross-dataset exports from raw tables."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare_research_data import candidates, fit_transformer, transform, FEATURES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(prepared, paths, output):
    prepared = Path(prepared)
    manifest = json.loads((prepared/'manifest.json').read_text())
    data, parts, states, sources = {}, {}, {}, {}
    for name, path in paths.items():
        if sha(path) != manifest['datasets'][name]['raw_sha256']:
            raise ValueError('Raw input hash mismatch: '+name)
        raw = pd.read_csv(path,low_memory=False)
        data[name] = candidates(raw,name)
        p = pd.read_csv(prepared/f'{name}_partitions.csv')
        if sha(prepared/f'{name}_partitions.csv') != manifest['datasets'][name]['partition_sha256']:
            raise ValueError('Partition hash mismatch')
        digest = sha(path)
        expected_ids = [f'{name}:{digest}:{i}' for i in range(len(raw))]
        if p.sample_id.tolist() != expected_ids:
            raise ValueError('Raw row alignment mismatch')
        label_col = 'URL_Type_obf_Type' if name=='iscx' else 'phishing'
        labels = raw[label_col].map({'benign':0,'phishing':1}) if name=='iscx' else raw[label_col]
        np.testing.assert_array_equal(labels.to_numpy(),p.label.to_numpy())
        parts[name] = p
        state = json.loads((prepared/f'{name}_transformer.json').read_text())
        recomputed = fit_transformer(data[name].loc[p.split=='train'])
        if recomputed != state:
            raise ValueError('Transformer differs from source-training-only reconstruction: '+name)
        states[name] = state
        sources[name] = {'raw_sha256':sha(path),'rows':len(raw),
            'non_numeric_predictor_columns':[c for c in raw if c!=label_col and not pd.api.types.is_numeric_dtype(raw[c])],
            'transformer_reconstructed_from_source_train':True,
            'raw_url_or_domain_identity_available':False}
    directions = []
    for source,target in [('iscx','mendeley'),('mendeley','iscx')]:
        filename=f'{source}_to_{target}_test.csv'
        path=prepared/filename
        if sha(path)!=manifest['exports'][filename]['sha256']:
            raise ValueError('Export hash mismatch')
        mask=parts[target].split=='test'
        expected=transform(data[target].loc[mask],states[source]).reset_index(drop=True)
        exported=pd.read_csv(path,float_precision='round_trip')
        pd.testing.assert_frame_equal(expected,exported[expected.columns],check_dtype=False,rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(exported.sample_id,parts[target].loc[mask,'sample_id'])
        np.testing.assert_array_equal(exported.label,parts[target].loc[mask,'label'])
        train_vectors=set(map(tuple,data[source].loc[parts[source].split=='train'].to_numpy()))
        target_vectors=list(map(tuple,data[target].loc[mask].to_numpy()))
        directions.append({'source':source,'target':target,'test_rows':len(exported),
            'export_sha256':sha(path),'source_fitted_export_reproduced':True,
            'outside_source_training_range':{c:int(((expected[c]<0)|(expected[c]>1)).sum()) for c in FEATURES},
            'target_rows_with_four_feature_vector_seen_in_source_train':sum(v in train_vectors for v in target_vectors),
            'collision_interpretation':'Numeric feature collisions are not evidence of duplicate URLs; URL/domain identity unavailable.'})
    result={'scope':'Read-only post-hoc protocol audit; no training, target tuning or feature correction.',
        'script_sha256':sha(__file__),'preparation_manifest_sha256':sha(prepared/'manifest.json'),
        'sources':sources,'directions':directions,
        'extraction_equivalence':'Not established by numeric reconstruction; requires original extractors and/or raw URLs.'}
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','iscx','mendeley','output']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    audit(a.prepared,{'iscx':a.iscx,'mendeley':a.mendeley},a.output)

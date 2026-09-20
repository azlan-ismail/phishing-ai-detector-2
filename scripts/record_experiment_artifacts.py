"""Inventory local completed experiment files without publishing their contents."""
import argparse
import hashlib
import json
from pathlib import Path


def inventory(work, output):
    work, output = Path(work), Path(output)
    folders = ['benchmark-smoke-v1','budget-pilot-v1','benchmark-five-v1','ablation-length-v1','ablation-gamma-zero-v1',
               'verified-five-v1','verified-ablation-length-v1','compared-ablation-length-v1',
               'verified-ablation-gamma-zero-v1','compared-ablation-gamma-zero-v1',
               'tuning-source-v1','benchmark-tuned-v1','verified-tuned-v1','compared-tuned-v1',
               'ablation-gamma-zero-tuned-v1','verified-gamma-zero-tuned-v1','compared-gamma-zero-tuned-v1',
               'source-convergence-v1','verified-source-convergence-v1']
    result = {'scope': 'Relative local artifact inventory; hashes are integrity records, not a claim of public availability. Raw data, predictions and weights remain local.', 'runs': {}}
    for name in folders:
        root = work / name
        if not root.is_dir():
            raise ValueError('Missing artifact directory: ' + name)
        records = []
        for path in sorted(root.rglob('*')):
            if path.is_file():
                records.append({'file': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        result['runs'][name] = {'file_count': len(records), 'files': records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: data['file_count'] for name, data in result['runs'].items()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    inventory(a.work, a.output)

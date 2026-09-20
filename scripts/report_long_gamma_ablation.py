"""Build descriptive tables from completed, verified matched-budget runs."""
import argparse
import json
from pathlib import Path
import pandas as pd


def report(baseline, ablation, comparison, output):
    baseline, ablation, comparison, output = map(Path, [baseline, ablation, comparison, output])
    for p in [baseline, ablation]:
        if not json.loads((p/'completion.json').read_text())['complete']:
            raise ValueError('Incomplete run')
    verified = json.loads((comparison/'comparison-verification.json').read_text())
    if verified['matched_prediction_files'] != 180:
        raise ValueError('Pairing verification missing')
    frames = [pd.read_csv(p/'metrics.csv') for p in [baseline, ablation]]
    lines = ['# Matched 8,000-update discount ablation\n',
        'Actual Python experiment comparing gamma 0 against the tuned gamma 0.99 benchmark. '
        'Both conditions use seeds 11, 23, 37, 51 and 71 on fixed candidate-grouped partitions. '
        'All neural models use 8,000 optimizer updates, learning rate .001 and batch size 64. '
        'The original source-validation selection is inherited unchanged; gamma zero is not independently tuned.\n',
        'Architecture, input features, preprocessing, data/replay/exploration seeds, reward weighting and '
        'supervised control settings remain matched. Thresholds are selected separately on source validation '
        'and then frozen for both tests. Thus threshold-dependent changes include source-threshold adaptation; '
        'AP provides a threshold-independent comparison.\n',
        'Entries are mean +/- sample SD across five training seeds. Paired changes are gamma 0 minus gamma 0.99. '
        'No significance or convergence claim is made.\n']
    for op, metrics in [('default',['average_precision','f1']),
                        ('validation_fpr_limit',['recall','fpr'])]:
        lines += ['\n## '+op+'\n', '| Source | Test | Model | Metric | Gamma .99 | Gamma 0 | Paired change |\n',
                  '|---|---|---|---|---:|---:|---:|\n']
        a,b = [d[(d.operating_point==op)&d.model.isin(['dqn','ddqn'])].set_index(['source','target','model','seed']).sort_index() for d in frames]
        delta = b[metrics]-a[metrics]
        for (source,target,model), group in a.groupby(level=['source','target','model']):
            key=(source,target,model)
            for metric in metrics:
                vals = [d.xs(key,level=['source','target','model'])[metric] for d in [a,b,delta]]
                cells = [f'{v.mean():.4f} +/- {v.std(ddof=1):.4f}' for v in vals]
                lines += ['| '+' | '.join([source,target,model,metric]+cells)+' |\n']
    lines += ['\n## Control checks\n',
        f"Matched {verified['matched_prediction_files']} validation/test prediction files by sample identity and label. "
        f"Checked {verified['non_rl_score_control_files_within_absolute_1e-12']} non-RL control files to absolute score tolerance 1e-12; "
        f"maximum score difference {verified['non_rl_max_absolute_score_difference']:.3g}; "
        f"changed operating-point decisions {verified['non_rl_changed_operating_point_decisions']}. "
        f"DQN/DDQN predictions and training histories at gamma zero match exactly in {verified['identical_gamma_zero_dqn_ddqn_files']} files. "
        'Actual neural optimizer updates and sample exposures match the tuned reference.\n',
        '\n## Evidence and limitations\n',
        'The [aggregate evidence](../research/results/gamma-zero-tuned/) includes every per-seed metric, '
        'mean/sample-SD summaries, paired changes, configuration and verification records. '
        'Local predictions, checkpoints, training histories and curve points are indexed in the '
        '[artifact inventory](../research/results/artifact-inventory.json). Original datasets are unchanged and remain local.\n',
        'This isolates gamma under inherited gamma-.99-selected settings; it does not compare independently optimized algorithms. '
        'Seed variation covers training randomness on fixed partitions. Prior target inspection informed the redesign, '
        'feature extraction equivalence is not fully established, and raw URL/domain identifiers are unavailable. '
        'The 8,000-update boundary was selected by earlier bounded tuning; completing this run does not establish convergence.\n']
    lines += ['\n## Reproduction\n',
        'Use the recorded Python environment and prepared exports, from the repository root. '
        'Keep completed output directories; use new directories for any rerun.\n',
        '```text\n'
        'python scripts/run_research_benchmark.py --prepared work/prepared-candidate-v2 --output work/ablation-gamma-zero-tuned-v1 --purpose exploratory --seeds 11 23 37 51 71 --tuning-selection work/tuning-source-v1/selection.json --discount-ablation-from-tuning --gamma 0 --trees 200 --threads 2\n'
        'python scripts/verify_benchmark.py --run work/ablation-gamma-zero-tuned-v1 --output work/verified-gamma-zero-tuned-v1\n'
        'python scripts/compare_gamma_ablation.py --baseline work/benchmark-tuned-v1 --ablation work/ablation-gamma-zero-tuned-v1 --output work/compared-gamma-zero-tuned-v1\n'
        'python scripts/report_long_gamma_ablation.py --baseline work/benchmark-tuned-v1 --ablation work/ablation-gamma-zero-tuned-v1 --comparison work/compared-gamma-zero-tuned-v1 --output docs/ablation-gamma-tuned-results.md\n'
        'python scripts/record_experiment_artifacts.py --work work --output research/results/artifact-inventory.json\n'
        '```\n']
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('\n'.join(lines).replace('|\n\n|', '|\n|'),encoding='utf-8')
    print(output)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','ablation','comparison','output']:
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    report(args.baseline,args.ablation,args.comparison,args.output)

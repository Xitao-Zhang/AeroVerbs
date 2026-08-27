import hashlib
import json
from collections import Counter
from pathlib import Path


project = Path('/root/AeroVerbs')
manifest_path = project / 'eval_manifests/batch_1.json'
batch_root = project / 'exp/batch_1'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
expected = [(s['skill'], s['scene_id']) for s in manifest['samples']]
assert len(expected) == 51 and len(set(expected)) == 51

all_results = {}
for model_name in ('bc', 'dagger'):
    output = batch_root / model_name
    lines = [line for line in (output / 'results.jsonl').read_text(encoding='utf-8').splitlines() if line]
    records = [json.loads(line) for line in lines]
    keys = [(r['skill'], r['scene_id']) for r in records]
    assert keys == expected
    assert len(keys) == len(set(keys)) == 51
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((output / 'run_config.json').read_text(encoding='utf-8'))
    assert summary['complete'] is True
    assert summary['processed_samples'] == summary['expected_samples'] == 51
    assert summary['missing_samples'] == 0
    assert config['rollout']['render'] is False
    assert config['rollout']['drone_radius'] == 0.15
    assert config['rollout']['max_steps'] == 200
    assert not list(output.rglob('*.png'))
    for record in records:
        assert record['model'] == model_name
        assert record['batch_id'] == 1
        assert record['status'] in {'success', 'collision', 'timeout', 'infrastructure_error'}
        trajectory_path = output / record['trajectory']
        trajectory = json.loads(trajectory_path.read_text(encoding='utf-8'))
        assert trajectory['semantic_label'] == record['skill']
        assert len(trajectory['points']) == record['trajectory_points']
        assert len(trajectory['points']) >= 2
    counts = Counter(r['status'] for r in records)
    assert dict(sorted(counts.items())) == summary['status_counts']
    assert counts['success'] == summary['success_count']
    actual_checkpoint_hash = hashlib.sha256(Path(config['checkpoint']).read_bytes()).hexdigest()
    assert actual_checkpoint_hash == config['checkpoint_sha256'] == summary['checkpoint_sha256']
    print(model_name, 'status_counts', dict(sorted(counts.items())))
    print(model_name, 'per_skill', {k: {'success': v['success'], 'expected': v['expected'], 'success_rate': v['success_rate']} for k, v in summary['per_skill'].items()})
    print(model_name, 'success_rate', summary['success_rate'])
    print(model_name, 'rollout_elapsed_s', summary['rollout_elapsed_s'])
    all_results[model_name] = {(r['skill'], r['scene_id']): r for r in records}

paired = Counter()
for key in expected:
    bc_ok = all_results['bc'][key]['status'] == 'success'
    dagger_ok = all_results['dagger'][key]['status'] == 'success'
    paired[(bc_ok, dagger_ok)] += 1
print('paired', {
    'both_success': paired[(True, True)],
    'bc_only_success': paired[(True, False)],
    'dagger_only_success': paired[(False, True)],
    'both_failed': paired[(False, False)],
})
print('png_count', len(list(batch_root.rglob('*.png'))))
print('trajectory_count', len(list(batch_root.rglob('trajectories/*.json'))))
print('AUDIT_OK')

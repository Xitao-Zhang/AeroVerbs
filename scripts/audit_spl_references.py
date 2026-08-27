import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

root = Path('eval_reference_paths')
with open(root / 'reference_index.json', encoding='utf-8') as f:
    index = json.load(f)
records = index['records']
assert index['processed_samples'] == 254
assert index['status_counts'] == {'valid': 254}
assert len(records) == 254
keys = [(r['skill'], r['scene_id']) for r in records]
assert len(set(keys)) == 254

skill_counts = Counter()
attempt_counts = Counter()
lengths = []
clearances = []
for record in records:
    assert record['status'] == 'valid'
    assert record['planner_success'] is True
    assert record['completion_validated'] is True
    assert record['minimum_authoritative_clearance_m'] + 1e-6 >= 0.15
    directory = root / record['output_relative_path']
    with np.load(directory / 'reference_path.npz') as archive:
        points = archive['points_std']
    with open(directory / 'trajectory.json', encoding='utf-8') as f:
        points_json = np.asarray(json.load(f)['points_std'], dtype=np.float32)
    with open(directory / 'metadata.json', encoding='utf-8') as f:
        metadata = json.load(f)
    assert points.ndim == 2 and points.shape[1] == 3 and len(points) >= 2
    assert np.all(np.isfinite(points))
    assert np.array_equal(points, points_json)
    assert metadata == record
    length = float(np.linalg.norm(np.diff(points.astype(np.float64), axis=0), axis=1).sum())
    assert abs(length - record['reference_path_length_m']) < 1e-6
    assert len(points) == record['reference_num_points']
    assert record['planner_config_hash'] == index['planner_config_hash']
    assert record['completion_config_hash'] == index['completion_config_hash']
    skill_counts[record['skill']] += 1
    attempt_counts[record['planner_diagnostics']['selected_attempt']] += 1
    lengths.append(length)
    clearances.append(record['minimum_authoritative_clearance_m'])

for path in [
    root / 'reference_index.json',
    Path('scripts/build_spl_reference_paths.py'),
    Path('uav_maneuver_planning/fast_planner/kinodynamic_astar.py'),
]:
    print('sha256', path, hashlib.sha256(path.read_bytes()).hexdigest())
print('records', len(records))
print('skill_counts', dict(sorted(skill_counts.items())))
print('attempt_counts', dict(sorted(attempt_counts.items())))
print('length_m', {'min': min(lengths), 'mean': sum(lengths)/len(lengths), 'max': max(lengths)})
print('minimum_clearance_m', min(clearances))
print('AUDIT_OK')

from pathlib import Path
import json,hashlib,sys
import numpy as np
B=Path(__file__).resolve().parents[1];R=B.parents[2]
sys.path.insert(0,str(R/'src'))
from build_point_graphs import fps
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
for lock in ['source_locked.json','input_locked.json']:
 for f,h in json.loads((B/lock).read_text()).items():assert sha(R/f)==h
for sp in ['train','val']:
 for f,h in json.loads((B/'targets'/sp/'locked.json').read_text()).items():assert sha(B/'targets'/sp/f)==h
m=json.loads((B/'manifest.json').read_text());checks=[]
for row in m['rows']:
 z=np.load(B/'samples'/(row['key']+'.npz'));a=np.load(row['target_source'])
 indices=fps(a['query_xyz'],2048) if len(a['query_xyz'])>2048 else np.arange(len(a['query_xyz']))
 assert np.array_equal(indices,z['causal_fps_indices'])
 for k in ['target','known','confidence']:assert np.array_equal(a[k][indices],z[k])
 assert np.all(a['confidence'][~a['known']]==0)
 pos=a['target'].astype(bool);neg=a['known']&~pos
 assert np.all(a['positive_support_count'][pos]>=2) and np.all(a['negative_support_count'][neg]>=2)
 assert not a['possible_inside_frames'][:,neg].any()
 assert np.all(z['target']<=z['known'])
 assert row['query_frame']<row['closed_command_start_frame']
 assert np.array_equal(z['input_rgb_pixel'],a['query_pixel'])
 assert np.array_equal(z['pixel'],a['query_pixel'][indices])
 assert np.array_equal(z['xyz_world'],a['query_xyz'][indices].astype(np.float32))
 assert np.isfinite(z['dino_feat']).all()
 checks.append({'key':row['key'],'positive':row['positive'],'negative':row['negative'],'unknown':row['unknown'],'query_strictly_before_close':True,'causal_sampling_verified':True})
train={r['episode'] for r in m['rows'] if r['split']=='train'};val={r['episode'] for r in m['rows'] if r['split']=='val'};assert not train&val
assert len(train)==m['train_episodes'] and len(val)==m['val_episodes']
(B/'audits/preflight.json').write_text(json.dumps({'checks':checks,'train_episodes':len(train),'val_episodes':len(val),'training_gate':m['training_gate_passed'],'source_locks_verified':True,'episode_disjoint':True},indent=2))
print('PREFLIGHT_PASSED',len(checks))

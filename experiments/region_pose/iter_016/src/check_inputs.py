from pathlib import Path
import sys,json,hashlib,numpy as np
B=Path(__file__).resolve().parents[1];R=B.parents[2]
sys.path.insert(0,str(R/'src'))
from build_point_graphs import fps
from supervision import targets
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
for lock in ['source_locked.json','input_locked.json']:
 for f,h in json.loads((B/lock).read_text()).items():assert sha(Path(f) if Path(f).is_absolute() else R/f)==h,f
m=json.loads((B/'manifest.json').read_text());checks=[]
for row in m['rows']:
 z=np.load(B/'samples'/(row['key']+'.npz'));a=np.load(row['target_source'])
 sel=fps(a['query_xyz'],2048) if len(a['query_xyz'])>2048 else np.arange(len(a['query_xyz']))
 assert np.array_equal(sel,z['causal_fps_indices'])
 y,k,c,idx=targets(row,a)
 for name,arr in [('target',y),('known',k),('confidence',c),('reference_source_index',idx)]:
  assert np.array_equal(arr[sel],z[name]),(row['key'],name)
 assert np.all(z['confidence'][~z['known']]==0)
 assert np.array_equal(z['input_rgb_pixel'],a['query_pixel'])
 assert np.array_equal(z['pixel'],a['query_pixel'][sel])
 assert np.array_equal(z['xyz_world'],a['query_xyz'][sel].astype(np.float32))
 assert np.isfinite(z['dino_feat']).all() and np.isfinite(z['pose']).all()
 assert row['query_frame']<row['closed_command_start_frame']
 assert len(np.unique(z['reference_source_index'][z['target']]))>=3
 assert len(np.unique(z['reference_source_index'][z['known']&~z['target']]))>=16
 checks.append({'key':row['key'],'offset':row['offset'],'fps_causal':True,'confidence_recomputed':True,'query_strict_preclose':True})
for sp in ['train','val']:
 for ev in {r['event'] for r in m['rows'] if r['split']==sp}:
  assert sorted(r['offset'] for r in m['rows'] if r['event']==ev)==[1,5,10,15]
tr={r['episode'] for r in m['rows'] if r['split']=='train'};va={r['episode'] for r in m['rows'] if r['split']=='val'};assert not tr&va
(B/'audits/preflight.json').write_text(json.dumps({'checks':checks,'train_events':len(tr),'val_events':len(va),'training_gate':m['training_gate_passed']},indent=2))
print('PREFLIGHT_PASSED',len(checks),flush=True)

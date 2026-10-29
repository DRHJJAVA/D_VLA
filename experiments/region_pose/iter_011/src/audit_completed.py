from pathlib import Path
import json,hashlib,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA');P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def anchor_ok(r):
 a=r['anchor_frame'];fits=r.get('bridge_results',[])
 if not fits or not r.get('valid'):return False
 if a==fits[0]['frame']:return bool(fits[0]['valid'])
 return any(x['frames'][1]==a and x['valid'] for x in r.get('time_comparisons',[]))
def main():
 assert json.loads((P/'stage_state.json').read_text())['stage']=='complete'
 for name in ['source_locked.json','input_locked.json']:
  for f,h in json.loads((P/name).read_text()).items():assert sha(Path(f) if Path(f).is_absolute() else P/f)==h
 frozen=json.loads((P/'pre_validation_freeze.json').read_text());assert frozen['train_summary_sha256']==sha(P/'train/summary.json')
 summary={};rows=[];missing=[]
 refs=json.loads((P.parent/'iter_003/evidence/base/protocol.json').read_text())['episodes']
 for split in ['train','val']:
  d=json.loads((P/split/'summary.json').read_text());base=json.loads((P.parent/'iter_008'/split/'summary.json').read_text());by={r['event']:r for r in base['cases']}
  for f,h in json.loads((P/split/'locked.json').read_text()).items():assert sha(P/split/f)==h
  for r in d['cases']:
   old=by[r['event']];assert r['query']==old['query'] and r['selected_held_frames']==old['selected_held_frames']
   if 'initial_corners' in r:assert r['initial_corners']==old['initial_corners']
   key=r['event']+'_query'+str(r['query']);sourcepixelcheck=None
   newfile=P/split/'trajectories'/(key+'.npz');oldfile=P.parent/'iter_008'/split/'correspondences'/(key+'.npz')
   if newfile.exists() and oldfile.exists():
    with np.load(newfile) as new,np.load(oldfile) as original:
     for fit in old.get('bridge_results',[]):
      t=str(fit['frame']);ids=original['track_id_'+t]
      assert np.array_equal(new['initial_uv'][ids],original['query_uv_'+t])
    sourcepixelcheck=True
   rows.append({'id':r['id'],'event':r['event'],'split':split,'query':r['query'],'klt_early':old['valid'],'cotracker_early':r['valid'],'klt_anchor':anchor_ok(old),'cotracker_anchor':anchor_ok(r),'same_source_pixels_when_available':sourcepixelcheck,'early_comparison':r.get('dual_bridge_difference'),'time_comparisons':r.get('time_comparisons',[]),'endpoints':[{k:v for k,v in x.items() if k!='query_world_to_anchor_hand'} for x in r.get('bridge_results',[])],'failure_reason':r.get('reason')})
  subset=[r for r in rows if r['split']==split];seen={r['id'] for r in subset}
  miss=[x['id'] for x in refs if x['split']==split and x['id'] not in seen]
  missing.extend({'id':x,'split':split,'reason':'no_003_held_fusion_case_unknown_not_failure'} for x in miss)
  summary[split]={'protocol_episodes':sum(x['split']==split for x in refs),'diagnosed_events':len(subset),'klt_early':sum(x['klt_early'] for x in subset),'cotracker_early':sum(x['cotracker_early'] for x in subset),'klt_anchor':sum(x['klt_anchor'] for x in subset),'cotracker_anchor':sum(x['cotracker_anchor'] for x in subset),'missing_episodes':miss}
 out={'summary':summary,'rows':rows,'missing':missing,'source_input_locks_pass':True,'validation_rules_frozen_after_training_diagnostic':True,'new_region_targets':False,'scoring_model_training':False,'test_read':False}
 (P/'completion_audit.json').write_text(json.dumps(out,indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

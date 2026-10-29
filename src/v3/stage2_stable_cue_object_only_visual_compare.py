"""After-lock visual-only comparison on train LOO and old validation.
Stable cue is automatic RGB-D evidence, not simulator contact or grasp truth.
No held-out reserved episodes touched.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
C=R/'data/v3/samples/stable_near_surface_cue_3dcc_v2'
NEW=R/'outputs/v3_stage2/stable_cue_object_only_transfer_3dcc_v2'
OLD=R/'outputs/v3_stage2/object_only_frequency_v1'
OUT=R/'audits/v3/stage2_stable_cue_object_only_visual_comparison.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(score,target,fraction):
 k=max(1,math.ceil(fraction*len(score)))
 ix=np.argpartition(score,-k)[-k:]
 return float(target[ix].sum()/max(target.sum(),1e-9))
def main():
 assert not OUT.exists()
 nlock=json.loads((NEW/'predictions_locked.json').read_text())
 olock=json.loads((OLD/'predictions_locked.json').read_text())
 assert nlock['status']=='stable_cue_object_only_3dcc_v2_predictions_locked'
 assert nlock['no_query_future'] and nlock['no_simulator_truth']
 rows=[]
 for split in ('train','val'):
  for id in S[split]:
   key=('train_loo' if split=='train' else split)+'/'+id
   npred=NEW/(key+'.npz');opred=OLD/(key+'.npz')
   assert sha(npred)==nlock['prediction_sha256'][key]
   assert sha(opred)==olock['prediction_sha256'][key]
   with np.load(npred) as z:ns=z['score'].copy();pixel=z['pixel'].copy()
   with np.load(opred) as z:os=z['score'].copy();assertpixel=z['pixel'].copy()
   with np.load(C/split/(id+'.npz')) as z:cue=z['cue'].copy();legacy=z['legacy_visual_target'].copy();cp=z['pixel'].copy()
   assert np.array_equal(pixel,cp) and np.array_equal(pixel,assertpixel)
   row=dict(id=id,split=split,points=len(cue),
    cue_mean=float(cue.mean()),legacy_mean=float(legacy.mean()))
   for fraction in (.1,.2,.3):
    key2='budget_'+str(round(fraction*100))
    row[key2]=dict(stable_transfer_to_stable_cue=metric(ns,cue,fraction),
     old_transfer_to_stable_cue=metric(os,cue,fraction),
     stable_transfer_to_old_cue=metric(ns,legacy,fraction),
     old_transfer_to_old_cue=metric(os,legacy,fraction),
     oracle_stable_cue=metric(cue,cue,fraction),
     random=math.ceil(fraction*len(cue))/len(cue))
   rows.append(row);print(json.dumps(row),flush=True)
 summary={}
 for split in ('train','val'):
  rr=[r for r in rows if r['split']==split]
  summary[split]={}
  for n in (10,20,30):
   k='budget_'+str(n)
   summary[split][k]={metric:float(np.mean([r[k][metric] for r in rr])) for metric in rr[0][k]}
 report=dict(status='visual_only_stable_cue_after_prediction_lock',
  new_prediction_lock_sha256=sha(NEW/'predictions_locked.json'),
  old_prediction_lock_sha256=sha(OLD/'predictions_locked.json'),
  no_simulator_truth=True,no_reserved_blind=True,
  reference='automatic narrow RGB-D interaction evidence, not grasp truth',
  caution='source cue may still highlight occluding robot or rim; transfer agreement cannot validate true contacts',
  summary=summary,rows=rows)
 OUT.write_text(json.dumps(report,indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

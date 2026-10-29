"""Visual-only comparison of strict object cutout versus prior local-context features.
Prediction files are locked before this audit. Validation future is reference only.
"""
from pathlib import Path
import hashlib,json,math,collections
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
SRC=R/'data/v3/samples/object_only_dino_v1'
NEW=R/'outputs/v3_stage2/object_only_frequency_v1'
OLD=R/'outputs/v3_stage2/visual_frequency_uniform448_3dcc'
POSE_OLD=R/'outputs/v3_stage2/visual_frequency_poseval16'
POSE_TARGET=R/'audits/v3/stage2_poseval16_visual_targets'
OUT=R/'audits/v3/stage2_object_only_visual_comparison.json'
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
M=json.loads((R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json').read_text())['rows']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(score,target,frac):
 k=max(1,int(math.ceil(frac*len(score))))
 top=np.argpartition(score,-k)[-k:]
 return float(target[top].sum()/max(target.sum(),1e-8))
def main():
 assert not OUT.exists()
 lock=json.loads((NEW/'predictions_locked.json').read_text())
 assert lock['status']=='strict_object_only_frequency_predictions_locked'
 assert lock['non_object_pixels_neutral'] and lock['no_query_future']
 rows=[]
 for split,ids in [('train_loo',S['train']),('val',S['val']),('poseval16',[e['id'] for e in M])]:
  for id in ids:
   npath=NEW/split/(id+'.npz')
   assert sha(npath)==lock['prediction_sha256'][split+'/'+id]
   old=POSE_OLD/'rank_1'/(id+'.npz') if split=='poseval16' else OLD/split/(id+'.npz')
   source=SRC/('train' if split=='train_loo' else split)/(id+'.npz')
   with np.load(npath) as z:new=z['score'].copy();pixel=z['pixel'].copy()
   with np.load(old) as z:prior=z['score'].copy();oldpixel=z['pixel'].copy()
   assert np.array_equal(pixel,oldpixel)
   if split=='poseval16':
    with np.load(POSE_TARGET/(id+'.npz')) as z:target=z['target'].copy()
   else:
    with np.load(source) as z:target=z['target'].copy()
   assert len(target)==len(new)==len(prior)
   row=dict(id=id,split=split,points=len(target))
   for frac in (.1,.2,.3):
    label='budget_'+str(int(100*frac))
    row[label]=dict(strict_object=metric(new,target,frac),
     prior_local_context=metric(prior,target,frac),
     random=math.ceil(frac*len(target))/len(target))
   rows.append(row);print(json.dumps(row),flush=True)
 summary={}
 for split in ('train_loo','val','poseval16'):
  rr=[x for x in rows if x['split']==split]
  summary[split]={}
  for frac in (10,20,30):
   key='budget_'+str(frac)
   summary[split][key]={name:float(np.mean([x[key][name] for x in rr]))
    for name in ('strict_object','prior_local_context','random')}
 report=dict(status='strict_single_object_visual_development_comparison',
  reference='automatic full-sequence visual target only; not grasp truth',
  new_prediction_lock_sha256=sha(NEW/'predictions_locked.json'),
  old_prediction_locks=[sha(OLD/'predictions_locked.json'),sha(POSE_OLD/'predictions_locked.json')],
  no_simulator_truth=True,summary=summary,rows=rows)
 OUT.write_text(json.dumps(report,indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

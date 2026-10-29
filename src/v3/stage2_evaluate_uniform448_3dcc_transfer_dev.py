"""Independent unreviewed old-contact diagnostic after uniform448 visual-score lock.
Development eight are not blind tests. No simulator truth is available to training."""
from pathlib import Path
import hashlib,json,math,sys
import h5py,numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_audit_preremoval_3d_visualcloud import project
M=R/'data/v3/manifests/stage2_localization_dev_8.json'
OUT=R/'outputs/v3_stage2/visual_frequency_uniform448_3dcc'
CONTACT=R/'data/v3/evaluation/localization_dev_contacts'
AUDIT=R/'audits/v3/stage2_uniform448_3dcc_transfer_dev_contact_audit.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lock=json.loads((OUT/'predictions_locked.json').read_text())
 assert lock['status']=='uniform448_3dcc_visual_frequency_locked_before_old_contact_audit'
 entries=json.loads(M.read_text())['rows'];rows=[]
 for e in entries:
  id=e['id'];g=e['asset_group'];ep=e['source_episode']
  pred=OUT/'dev'/(id+'.npz')
  assert sha(pred)==lock['prediction_sha256']['dev/'+id]
  with np.load(pred) as z:score=z['score'].copy();pts=z['xyz'].copy();box=z['source_box'].copy()
  state=json.loads((CONTACT/g/(ep+'_states_contacts.json')).read_text())['rows']
  r0=state[0];p0=np.asarray(r0['object_position']);rot0=np.asarray(r0['object_rotation'])
  local=[]
  for row in state:
   p=np.asarray(row['object_position']);rot=np.asarray(row['object_rotation'])
   for c in row['contacts']:local.append((np.asarray(c['world'])-p)@rot)
  world=np.asarray(local)@rot0.T+p0
  with h5py.File(e['rgbd_path']) as f:
   depth=f['agentview/depth_m'][0];K=f['agentview/K'][0];T=f['agentview/camera_to_world'][0]
  u,v,visible=project(world,K,T,depth,.02)
  x1,y1,x2,y2=box;visible&=(u>=x1)&(u<x2)&(v>=y1)&(v<y2)
  cp=world[visible]
  if len(cp):
   near,ix=cKDTree(pts).query(cp);ix=ix[near<.015]
  else:ix=np.empty(0,int)
  r=dict(id=id,asset_group=g,visual_points=len(pts),visible_contacts=len(cp),evaluable_contacts=len(ix))
  if len(ix):
   for fraction in (.1,.2,.3):
    k=max(1,math.ceil(fraction*len(score)));top=np.argpartition(score,-k)[-k:]
    r['budget_'+str(int(100*fraction))]=dict(transfer=float(np.isin(ix,top).mean()),random=k/len(score))
  rows.append(r);print(json.dumps(r),flush=True)
 summary={}
 for budget in (10,20,30):
  rr=[x['budget_'+str(budget)] for x in rows if 'budget_'+str(budget) in x]
  summary[str(budget)]={k:float(np.mean([q[k] for q in rr])) for k in ('transfer','random')}
 summary['evaluable']=sum('budget_20' in x for x in rows)
 if AUDIT.exists():raise FileExistsError(AUDIT)
 AUDIT.write_text(json.dumps(dict(status='development_evaluation_only',old_simulator_contacts='unreviewed; no training',prediction_lock_sha256=sha(OUT/'predictions_locked.json'),summary=summary,rows=rows),indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

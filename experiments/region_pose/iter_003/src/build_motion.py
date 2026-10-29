"""003: frozen finite-finger-volume evidence gate. Reuses immutable RGB-D measurements, not simulator truth."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
I=Path(__file__).resolve().parents[1];OLD=I.parent/'iter_002';M=I/'evidence/motion'
def runs(mask):
 x=np.flatnonzero(mask)
 return [a for a in np.split(x,np.flatnonzero(np.diff(x)>1)+1) if len(a)>=5] if len(x) else []
def main():
 protocol=json.loads((I/'evidence/base/protocol.json').read_text());summary=[];audit=[];source={}
 for row in protocol['episodes']:
  id=row['id'];jf=OLD/'evidence/motion'/(id+'.json');nf=jf.with_suffix('.npz')
  v=json.loads(jf.read_text());source[str(jf)]=hashlib.sha256(jf.read_bytes()).hexdigest();source[str(nf)]=hashlib.sha256(nf.read_bytes()).hexdigest()
  with np.load(nf) as z:
   H=z['hand_to_world'];q=z['measured_finger_q'];ov=z['visual_overlap'];pts=z['points_world'];offset=z['frame_offsets']
   n=len(q);counts=np.zeros(n,int);sides=np.zeros((n,2),int)
   for t in range(1,n):
    p=pts[offset[t]:offset[t+1]];ph=(p-H[t,:3,3])@H[t,:3,:3]
    inside=(abs(ph[:,0])<=.008)&(ph[:,2]>=.0854)&(ph[:,2]<=.1014)&(ph[:,1]>=q[t,1]+.0005)&(ph[:,1]<=q[t,0]-.0005)
    if inside.any():
     pp=ph[inside];counts[t]=len(np.unique(np.rint(pp/.001).astype(np.int64),axis=0))
     # provenance-side counts are diagnostic only, not a two-visible-sides requirement
     side=z['point_side'][offset[t]:offset[t+1]][inside]
     sides[t]=[(side==s).sum() for s in [0,1]]
   gap=q[:,0]-q[:,1]-.001
   cloudn=np.array([r['hand_frame_object_points'] for r in v['frame_diagnostics']])
   for a in v['attempts']:
    ix=np.arange(n);start=a['start_frame'];end=a['end_frame']
    valid=(ix>=start)&(ix<=end)&(gap>0)&((H[:,2,3]-H[start,2,3])>.025)&(ov>=.5)&(cloudn>=20)&(counts>=3)
    spans=runs(valid);accepted=np.concatenate(spans) if spans else np.empty(0,int)
    oldstatus=a['status'];a['status']='held_candidate_with_visual_support' if len(accepted) else 'unresolved_no_sustained_finger_volume_evidence'
    a['moving_visual_support_frames']=accepted.tolist();a['accepted_contact_frames']=accepted.tolist()
    a['evidence_runs']=[s.tolist() for s in spans];a['old_iteration_002_status']=oldstatus
    a['whole_task_success_claim']=False
    audit.append(dict(id=id,split=row['split'],attempt=a['attempt'],old_status=oldstatus,new_status=a['status'],eligible_frames=len(accepted),max_consecutive=max([len(x) for x in spans],default=0),valid_frames_before_consecutive_gate=int(valid.sum()),median_region_voxels=float(np.median(counts[accepted])) if len(accepted) else 0,frames=accepted.tolist()))
   for t,r in enumerate(v['frame_diagnostics']):
    r['finite_finger_region_observed_voxels']=int(counts[t]);r['finite_finger_region_source_side_counts']=sides[t].tolist()
  (M/(id+'.json')).write_text(json.dumps(v,indent=2))
  shutil.copy2(nf,M/nf.name)
  summary.append({k:v[k] for k in ['id','split','frames','attempts']})
  print(json.dumps(audit[-len(v['attempts']):]),flush=True)
 controls=[r for r in audit if (r['id'],r['attempt']) in [('alphabet_soup_demo_1',1),('tomato_sauce_demo_3',1)]]
 assert len(controls)==2
 passed=all(r['eligible_frames']==0 for r in controls)
 (I/'audits/motion_gate_diagnostic.json').write_text(json.dumps(dict(rows=audit,controls=controls,empty_controls_not_promoted=passed,source_sha256=source),indent=2))
 assert passed,'Empty attempt control promoted; freeze downstream and inspect. No threshold tuning.'
 (M/'protocol.json').write_text(json.dumps(dict(iteration='003',gap_rule='positive geometry only',finite_finger_region_min_unique_1mm_voxels=3,min_consecutive_frames=5,other_gates='lift25mm, existing local cloud20pts, overlap50pct within3mm',source_iteration='002',classification_is_candidate_not_ground_truth=True),indent=2))
 (M/'summary.json').write_text(json.dumps(dict(episodes=summary),indent=2))
 print('motion gate diagnostic complete; controls passed',flush=True)
if __name__=='__main__':main()

"""Rank visual-change candidates by wrist-camera approach in calibrated RGB-D.
A technical name-free localization pilot; scores are NOT grasp-region predictions.
Only provided RGB-D and camera calibration are read, never actions or region truth."""
from pathlib import Path
import json,h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
def run(entry):
 source=Path(entry['rgbd_path'])
 proposal=R/'data/v3/proposals/motion_only'/entry['asset_group']/(source.stem+'.json')
 q=json.loads(proposal.read_text())
 assert q['status']=='complete' and len(q['candidates'])>=2
 output=R/'data/v3/proposals/motion_rgbd_rerank'/entry['asset_group']/(source.stem+'.json')
 if output.exists():
  old=json.loads(output.read_text())
  assert old['status']=='complete' and old['source_rgbd']==str(source)
  return old
 with h5py.File(source,'r') as f:
  depth=f['agentview/depth_m'][0]
  K=f['agentview/K'][0]
  T=f['agentview/camera_to_world'][0]
  wrist=f['robot0_eye_in_hand/camera_to_world'][:,:3,3]
 scores=[]
 for cand in q['candidates'][:2]:
  x1,y1,x2,y2=cand['box'];u=(x1+x2)/2;v=(y1+y2)/2
  z=float(np.median(depth[y1:y2,x1:x2]))
  cam=np.array([(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z])
  world=T[:3,:3]@cam+T[:3,3]
  trajectory=np.linalg.norm(wrist-world,axis=1)
  initial=float(trajectory[0])
  closest=float(np.min(trajectory[max(1,len(trajectory)//10):]))
  approach=max(0.,initial-closest)*float(np.exp(-closest/.2))
  scores.append(dict(box=cand['box'],visual_change_score=cand['score'],initial_distance_m=initial,closest_later_distance_m=closest,approach_score=approach,window=cand['window']))
 selected=int(np.argmax([z['approach_score'] for z in scores]))
 result=dict(status='complete',id=entry['id'],source_rgbd=str(source),frames=entry['frames'],method='first-frame RGB-D candidate world point and full wrist-camera pose trajectory',uses_object_name=False,uses_text_prompt=False,uses_actions=False,uses_simulator_truth=False,uses_region_labels=False,selected_index=selected,selected=scores[selected],candidates=scores,original_candidate_count=q['candidate_count'],note='localization heuristic; not a learned grasp-region model')
 output.parent.mkdir(parents=True,exist_ok=True)
 tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(output)
 print(json.dumps(dict(id=entry['id'],selected_index=selected,approach_scores=[round(z['approach_score'],4) for z in scores])),flush=True)
 return result
def main():
 entries=[json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]
 for e in entries:run(e)
if __name__=='__main__':main()

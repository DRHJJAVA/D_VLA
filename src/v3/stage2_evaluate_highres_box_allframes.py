"""Evaluation-only all-contact-frame audit for A_highres; no reference enters training."""
from pathlib import Path
import argparse,json,math
import h5py,numpy as np,torch
from stage2_highres_box_teacher import R,OUTPUT,load,BoxTeacher
from stage2_evaluate_teacher import project
def patches(u,v,vis,crop):
 x1,y1,x2,y2=crop
 ids=np.flatnonzero(vis)
 ids=ids[(u[ids]>=x1)&(u[ids]<x2)&(v[ids]>=y1)&(v[ids]<y2)]
 return set((max(0,min(31,int((v[i]-y1)*32/(y2-y1)))),max(0,min(31,int((u[i]-x1)*32/(x2-x1))))) for i in ids)
def episode(net,e):
 data,meta=load(e,'cpu')
 with torch.no_grad():_,scores=net(*data)
 scores=scores.numpy()
 asset=e['asset_group'];ep=e['source_episode']
 state=json.loads((R/'data/v2/stage1'/asset/(ep+'_states_contacts.json')).read_text())
 with np.load(R/'data/v2/stage1'/asset/'surface.npz') as mesh:local=mesh['points_object'][::10]
 frame_ids=sorted({i for c in state['candidates'] if c['numeric_pass'] for i in c['window'] if state['rows'][i]['contacts']})
 results=[]
 with h5py.File(e['rgbd_path'],'r') as f:
  g=f['agentview']
  for frame in frame_ids:
   row=state['rows'][frame]
   contacts=np.asarray([c['world'] for c in row['contacts']],dtype=float).reshape(-1,3)
   world=local@np.asarray(row['object_rotation']).T+np.asarray(row['object_position'])
   depth=g['depth_m'][frame];K=g['K'][frame];T=g['camera_to_world'][frame]
   ou,ov,ovis=project(world,K,T,depth,.012);cu,cv,cvis=project(contacts,K,T,depth,.02)
   obj=patches(ou,ov,ovis,meta['crop']);ref=patches(cu,cv,cvis,meta['crop'])&obj
   if not obj or not ref:continue
   k=max(1,math.ceil(.2*len(obj)))
   rank=sorted(obj,key=lambda p:float(scores[frame,p[0],p[1]]),reverse=True)
   cover=len(set(rank[:k])&ref)/len(ref)
   results.append(dict(frame=frame,visible_object_patches=len(obj),evaluable_contact_patches=len(ref),coverage=cover,random_budget=k/len(obj)))
 return dict(id=e['id'],contact_positive_frames=len(frame_ids),evaluable_frames=len(results),mean_coverage=float(np.mean([r['coverage'] for r in results])) if results else None,mean_random_budget=float(np.mean([r['random_budget'] for r in results])) if results else None,total_evaluable_contact_patches=sum(r['evaluable_contact_patches'] for r in results),frames=results)
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],default='val');a=p.parse_args()
 assert (OUTPUT/'done.json').exists()
 torch.set_num_threads(4)
 saved=torch.load(OUTPUT/'best.pt',map_location='cpu',weights_only=False)
 net=BoxTeacher().cpu().eval();net.load_state_dict(saved['state'])
 rows={x['id']:x for x in [json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]}
 split=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
 result=[episode(net,rows[id]) for id in split[a.split]]
 valid=[r for r in result if r['evaluable_frames']]
 out=dict(status='evaluation_only',variant='A_highres_box',partition=a.split,checkpoint_epoch=saved['epoch'],independent_episodes=len(valid),mean_episode_coverage=float(np.mean([r['mean_coverage'] for r in valid])) if valid else None,mean_episode_random_budget=float(np.mean([r['mean_random_budget'] for r in valid])) if valid else None,contact_positive_frames=sum(r['contact_positive_frames'] for r in result),evaluable_frames=sum(r['evaluable_frames'] for r in result),rows=result,note='same episode contains correlated frames; old contacts unreviewed; not physical success')
 target=OUTPUT/'review';target.mkdir(exist_ok=True)
 (target/('all_contact_frames_'+a.split+'.json')).write_text(json.dumps(out,indent=2,ensure_ascii=False))
 print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
 for r in result:print(json.dumps({k:v for k,v in r.items() if k!='frames'}),flush=True)
if __name__=='__main__':main()

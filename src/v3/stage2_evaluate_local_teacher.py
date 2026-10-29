"""Independent local-crop teacher diagnostic. Old simulator contacts are evaluation-only."""
from pathlib import Path
import argparse,json,math
import h5py,numpy as np,torch
from stage2_local_teacher import R,OUTPUT,load,Teacher
from stage2_evaluate_teacher import project
def case(net,e,device):
 data,meta=load(e,device)
 with torch.no_grad():_,score=net(*data)
 attn=score.cpu().numpy()
 asset=e['asset_group'];ep=e['source_episode']
 state=json.loads((R/'data/v2/stage1'/asset/(ep+'_states_contacts.json')).read_text())
 candidates=[x for x in state['candidates'] if x['numeric_pass']]
 assert candidates
 window=max(candidates,key=lambda x:x['end']-x['start'])['window']
 frame=max(window,key=lambda t:len(state['rows'][t]['contacts']))
 contacts=np.asarray([c['world'] for c in state['rows'][frame]['contacts']],dtype=float).reshape(-1,3)
 pose=state['rows'][frame]
 with h5py.File(e['rgbd_path'],'r') as f:
  g=f['agentview'];rgb=g['rgb'][frame];depth=g['depth_m'][frame];K=g['K'][frame];T=g['camera_to_world'][frame]
 cu,cv,cvis=project(contacts,K,T,depth)
 with np.load(R/'data/v2/stage1'/asset/'surface.npz') as mesh:obj=mesh['points_object'][::10]
 world=obj@np.asarray(pose['object_rotation']).T+np.asarray(pose['object_position'])
 ou,ov,ovis=project(world,K,T,depth,.012)
 x1,y1,x2,y2=meta['crop']
 def patches(u,v,vis):
  ids=np.flatnonzero(vis)
  ids=ids[(u[ids]>=x1)&(u[ids]<x2)&(v[ids]>=y1)&(v[ids]<y2)]
  return set((max(0,min(31,int((v[i]-y1)*32/(y2-y1)))),max(0,min(31,int((u[i]-x1)*32/(x2-x1))))) for i in ids)
 object_patches=patches(ou,ov,ovis)
 contact_patches=patches(cu,cv,cvis)
 applicable=contact_patches&object_patches
 m=len(object_patches);k=max(1,math.ceil(.2*m)) if m else 0
 rank=sorted(object_patches,key=lambda p:float(attn[frame,p[0],p[1]]),reverse=True)
 top=set(rank[:k])
 return dict(id=e['id'],frame=frame,source_box=meta['source_box'],crop=meta['crop'],visible_object_patches=m,evaluable_contact_patches=len(applicable),coverage_at_top20pct_visible_object_patches=len(top&applicable)/len(applicable) if applicable else None,random_budget_baseline=k/m if m else None,object_attention_mass=float(sum(attn[frame,p[0],p[1]] for p in object_patches)),note='old unreviewed simulator contacts; evaluation only')
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
 assert (OUTPUT/'done.json').exists(),'training incomplete'
 torch.set_num_threads(4);saved=torch.load(OUTPUT/'best.pt',map_location='cpu',weights_only=False)
 net=Teacher('A').to(a.device).eval();net.load_state_dict(saved['state'])
 rows={x['id']:x for x in [json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]}
 split=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
 result=[case(net,rows[id],a.device) for id in split[a.split]]
 values=[x['coverage_at_top20pct_visible_object_patches'] for x in result if x['coverage_at_top20pct_visible_object_patches'] is not None]
 baseline=[x['random_budget_baseline'] for x in result if x['coverage_at_top20pct_visible_object_patches'] is not None]
 out=dict(status='evaluation_only',variant='A_local',partition=a.split,checkpoint_epoch=saved['epoch'],evaluable=len(values),mean_coverage=float(np.mean(values)) if values else None,mean_random_budget=float(np.mean(baseline)) if baseline else None,rows=result)
 target=OUTPUT/'review';target.mkdir(exist_ok=True)
 (target/('diagnostic_'+a.split+'.json')).write_text(json.dumps(out,indent=2,ensure_ascii=False))
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()

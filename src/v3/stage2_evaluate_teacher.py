"""Independent diagnostic for Stage-2 teacher. Old simulation contacts/surfaces are evaluation-only."""
from pathlib import Path
import argparse,json,math,sys
import h5py,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stage2_crossview_teacher import R,Teacher,load_case
def project(world,K,T,depth,tol=.02):
 local=(world-T[:3,3])@T[:3,:3];z=local[:,2]
 safe=np.where(z>0,z,1)
 u=np.rint(K[0,0]*local[:,0]/safe+K[0,2]).astype(int)
 v=np.rint(K[1,1]*local[:,1]/safe+K[1,2]).astype(int)
 good=(z>0)&(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0])
 ids=np.flatnonzero(good);visible=np.zeros(len(world),dtype=bool)
 if len(ids):visible[ids]=np.abs(depth[v[ids],u[ids]]-z[ids])<tol
 return u,v,visible
def case(net,entry,out,plot=False):
 asset=entry['asset_group'];ep=entry['source_episode']
 with torch.no_grad():
  data=load_case(entry,'cuda',net.variant!='A')
  _,scores=net(*data)
  attention=scores.cpu().numpy()
 state=json.loads((R/'data/v2/stage1'/asset/(ep+'_states_contacts.json')).read_text())
 candidates=[x for x in state['candidates'] if x['numeric_pass']]
 assert candidates
 window=max(candidates,key=lambda x:x['end']-x['start'])['window']
 frame=max(window,key=lambda t:len(state['rows'][t]['contacts']))
 contacts=np.asarray([c['world'] for c in state['rows'][frame]['contacts']],dtype=float).reshape(-1,3)
 pose=state['rows'][frame]
 with h5py.File(entry['rgbd_path'],'r') as f:
  g=f['agentview'];rgb=g['rgb'][frame];depth=g['depth_m'][frame];K=g['K'][frame];T=g['camera_to_world'][frame]
  wrist=f['robot0_eye_in_hand/rgb'][frame]
 u,v,visible=project(contacts,K,T,depth)
 contact_patches=set((int(v[i]*32/depth.shape[0]),int(u[i]*32/depth.shape[1])) for i in np.flatnonzero(visible))
 with np.load(R/'data/v2/stage1'/asset/'surface.npz') as mesh:
  obj=mesh['points_object'][::10]
 world=obj@np.asarray(pose['object_rotation']).T+np.asarray(pose['object_position'])
 su,sv,sviz=project(world,K,T,depth,.012)
 object_patches=set((int(sv[i]*32/depth.shape[0]),int(su[i]*32/depth.shape[1])) for i in np.flatnonzero(sviz))
 applicable=contact_patches&object_patches
 m=len(object_patches);k=max(1,math.ceil(.2*m)) if m else 0
 score=attention[frame]
 rank=sorted(object_patches,key=lambda p:float(score[p]),reverse=True)
 top=set(rank[:k]);coverage=len(applicable&top)/len(applicable) if applicable else None
 baseline=k/m if m else None
 mass=float(sum(score[p] for p in object_patches))
 res=dict(id=entry['id'],frame=frame,candidate_contact_count=len(contacts),visible_contact_patches=len(contact_patches),visible_object_patches=m,evaluable_contact_patches=len(applicable),coverage_at_top20pct_visible_object_patches=coverage,random_budget_baseline=baseline,object_attention_mass=mass,attention_entropy=float(-(score*np.log(score+1e-12)).sum()/np.log(score.size)),scope='evaluation-only provisional contacts; not proof of enclosed-region prediction')
 if plot:
  fig,ax=plt.subplots(1,4,figsize=(16,4),layout='constrained')
  with h5py.File(entry['rgbd_path'],'r') as f:
   start=f['agentview/rgb'][0]
  for a,img,title in zip(ax[:3],[start,rgb,wrist],['Supplied sequence: first frame','Fixed camera at evaluated frame','Wrist camera at same frame']):
   a.imshow(img);a.set_title(title,fontsize=10);a.axis('off')
  ax[3].imshow(rgb)
  heat=ax[3].imshow(score,extent=(0,rgb.shape[1],rgb.shape[0],0),interpolation='nearest',cmap='inferno',alpha=.62,vmin=0,vmax=float(np.quantile(score,.995)))
  ax[3].set_title('Learned attention, relative support only',fontsize=10);ax[3].axis('off')
  fig.colorbar(heat,ax=ax[3],fraction=.046,pad=.04)
  fig.suptitle(entry['id']+' | frame '+str(frame)+' | model output, no manual region labels')
  path=out/(entry['id']+'.jpg');fig.savefig(path,dpi=130,facecolor='white');plt.close(fig)
  res['preview']=str(path)
 return res
def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',default='A');p.add_argument('--seed',type=int,default=2026);a=p.parse_args()
 torch.set_num_threads(4)
 ck=R/'outputs/v3_stage2'/a.variant/f'seed_{a.seed}'/'best.pt'
 saved=torch.load(ck,map_location='cpu',weights_only=False)
 net=Teacher(a.variant).cuda().eval();net.load_state_dict(saved['state'])
 entries={x['id']:x for x in [json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]}
 split=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
 out=R/'outputs/v3_stage2'/a.variant/f'seed_{a.seed}'/'review';out.mkdir(exist_ok=True)
 report=dict(variant=a.variant,seed=a.seed,checkpoint_epoch=saved['epoch'],partition={},note='Automated contact records are independent evaluation only and still need visual review; attention is not calibrated probability')
 for part in ('val','test'):
  rows=[]
  for name in split[part]:
   x=case(net,entries[name],out,plot=True);rows.append(x);print(json.dumps(x),flush=True)
  numbers=[x['coverage_at_top20pct_visible_object_patches'] for x in rows if x['coverage_at_top20pct_visible_object_patches'] is not None]
  report['partition'][part]=dict(count=len(rows),evaluable=len(numbers),mean_coverage=float(np.mean(numbers)) if numbers else None,rows=rows)
 (out/'diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='rows'} for k,v in report['partition'].items()}),flush=True)
if __name__=='__main__':main()

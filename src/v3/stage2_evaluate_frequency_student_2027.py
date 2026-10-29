"""Independent old-contact diagnostic for an image-only PointNet++ student; no training here."""
from pathlib import Path
import argparse,json,sys,math
import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pointnet2_grasp import PointNet2Grasp
from stage2_train_frequency_student import R,DATA,GRAPH_DATA,S,tensor
from stage2_audit_preremoval_3d_visualcloud import project
M={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
def calc(id,split,net,device):
 e=M[id];aid=e['asset_group'];ep=e['source_episode']
 folder='test_inference' if split=='test' else split
 path=DATA/folder/(id+'.npz');graph=GRAPH_DATA/'graphs'/folder/(id+'.npz')
 with np.load(path) as f:s={k:f[k].copy() for k in f.files if k in ('xyz','dino_feat','target','pixel','source_box')}
 has_teacher='target' in s
 if not has_teacher:s['target']=np.zeros(len(s['xyz']),dtype=np.float32)
 with np.load(graph) as f:g={k:f[k].copy() for k in f.files}
 with torch.inference_mode():
  xyz,feat,_,gg=tensor((id,s,g),device)
  pred=net(xyz,feat,gg).detach().cpu().numpy()
 state=json.loads((R/'data/v2/stage1'/aid/(ep+'_states_contacts.json')).read_text())
 r0=state['rows'][0];p0=np.asarray(r0['object_position']);rot0=np.asarray(r0['object_rotation'])
 local=[]
 for row in state['rows']:
  p=np.asarray(row['object_position']);rot=np.asarray(row['object_rotation'])
  for c in row['contacts']:local.append((np.asarray(c['world'])-p)@rot)
 if not local:return dict(id=id,reason='no reference contacts')
 contacts=np.asarray(local)@rot0.T+p0
 with h5py.File(e['rgbd_path']) as f:
  depth=f['agentview/depth_m'][0];K=f['agentview/K'][0];T=f['agentview/camera_to_world'][0]
 u,v,visible=project(contacts,K,T,depth,.02)
 x1,y1,x2,y2=s['source_box']
 visible&=(u>=x1)&(u<x2)&(v>=y1)&(v<y2)
 cp=contacts[visible]
 if not len(cp):return dict(id=id,reason='no visible reference contacts')
 nearest,ix=cKDTree(s['xyz']).query(cp)
 ix=ix[nearest<.015]
 if not len(ix):return dict(id=id,reason='no near-cloud reference contacts')
 pts=s['xyz'];pixel=s['pixel'];teacher=s['target']
 center3d=-np.linalg.norm(pts-pts.mean(0),axis=1)
 center2d=-np.linalg.norm(pixel-np.asarray([(x1+x2)/2,(y1+y2)/2]),axis=1)
 result=dict(id=id,visible_reference_contacts=len(cp),evaluable_reference_contacts=len(ix),visual_points=len(pts))
 for frac in (.1,.2,.3):
  budget=max(1,math.ceil(frac*len(pts)))
  result[f'budget_{int(frac*100)}']={'random':budget/len(pts)}
  for name,score in [('student',pred),('center3d',center3d),('center2d',center2d)]+([('train_demo_frequency',teacher)] if has_teacher else []):
   top=np.argpartition(score,-budget)[-budget:]
   result[f'budget_{int(frac*100)}'][name]=float(np.isin(ix,top).mean())
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=('val',),default='val');a=p.parse_args()
 out=R/'outputs/v3_stage2/frequency_student/seed_2027'
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
 device='cpu';net=PointNet2Grasp().to(device);net.load_state_dict(ck['state']);net.eval()
 rows=[]
 for id in S[a.split]:
  r=calc(id,a.split,net,device);rows.append(r);print(json.dumps(r),flush=True)
 summary={}
 for frac in (10,20,30):
  rr=[r[f'budget_{frac}'] for r in rows if f'budget_{frac}' in r]
  summary[str(frac)]={k:float(np.mean([x[k] for x in rr])) for k in ('student','center3d','center2d','random')+ (('train_demo_frequency',) if a.split=='val' else ())} if rr else {}
 summary['evaluable_episodes']=sum('budget_20' in x for x in rows)
 summary['total_episodes']=len(rows)
 report=dict(status='provisional_contact_diagnostic',split=a.split,checkpoint_epoch=ck['epoch'],checkpoint_selection='validation train-demo frequency target, not old contacts',old_simulator_contact='evaluation only; unreviewed',scope='visible first-pose contacts only; enclosure not measured',summary=summary,rows=rows)
 target=out/'review'/('student_contact_'+a.split+'.json');target.parent.mkdir(exist_ok=True)
 target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

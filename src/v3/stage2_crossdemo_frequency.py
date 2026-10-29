"""Name-free cross-demonstration frequency baseline using TRAIN visual interaction cues.
Training/retrieval only sees RGB-D-derived point samples and frozen DINO features.
Old simulator contacts and pose are opened only inside the evaluation function."""
from pathlib import Path
import json,math,sys
import h5py,numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage2_audit_preremoval_3d_visualcloud import project
R=Path('/mnt/nvme1/drh/D_VLA')
DATA=R/'data/v3/samples/visual_interaction_v1'
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
E={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
def load(split,id):
 with np.load(DATA/split/(id+'.npz')) as f:
  d={k:f[k].copy() for k in ('xyz','dino_feat','pixel','source_box','target')}
 d['id']=id;d['split']=split
 embedding=d['dino_feat'].mean(0)
 d['embedding']=embedding/np.linalg.norm(embedding)
 return d
def uv(d):
 x1,y1,x2,y2=d['source_box']
 x=np.clip((d['pixel'][:,0]-x1)/max(1,x2-x1),0,.9999)
 y=np.clip((d['pixel'][:,1]-y1)/max(1,y2-y1),0,.9999)
 return x,y
def visual_maps(d):
 x,y=uv(d);ix=(x*32).astype(int);iy=(y*32).astype(int)
 maps=[]
 for target in (d['target'],(d['target']>=np.quantile(d['target'],.8)).astype(float)):
  numerator=np.zeros((32,32));count=np.zeros((32,32))
  np.add.at(numerator,(iy,ix),target);np.add.at(count,(iy,ix),1)
  numerator=gaussian_filter(numerator,1.0)
  count=gaussian_filter(count,1.0)
  maps.append(np.divide(numerator,count,out=np.zeros_like(numerator),where=count>1e-6))
 return maps
def rank(query,train):
 possible=[d for d in train if d['id']!=query['id']]
 similarities=np.asarray([float(query['embedding']@d['embedding']) for d in possible])
 ix=np.argsort(-similarities)[:5]
 weights=np.exp((similarities[ix]-similarities[ix].max())/.03)
 weights/=weights.sum()
 selected=[possible[j] for j in ix]
 fused=[]
 for method in range(2):
  grid=sum(w*d['maps'][method] for w,d in zip(weights,selected))
  x,y=uv(query)
  fused.append(map_coordinates(grid,np.stack((y*31,x*31)),order=1,mode='nearest'))
 return fused,[(d['id'],float(similarities[j]),float(w)) for j,d,w in zip(ix,selected,weights)]
def contact_indices(d):
 # Evaluation boundary: all below is old simulator truth, never retrieved for map creation.
 e=E[d['id']];aid=e['asset_group'];ep=e['source_episode']
 state=json.loads((R/'data/v2/stage1'/aid/(ep+'_states_contacts.json')).read_text())
 r0=state['rows'][0];p0=np.asarray(r0['object_position']);rot0=np.asarray(r0['object_rotation'])
 local=[]
 for row in state['rows']:
  p=np.asarray(row['object_position']);rot=np.asarray(row['object_rotation'])
  for c in row['contacts']:local.append((np.asarray(c['world'])-p)@rot)
 if not local:return np.empty(0,dtype=int)
 world=np.asarray(local)@rot0.T+p0
 with h5py.File(e['rgbd_path']) as f:
  depth=f['agentview/depth_m'][0];K=f['agentview/K'][0];T=f['agentview/camera_to_world'][0]
 u,v,visible=project(world,K,T,depth,.02)
 x1,y1,x2,y2=d['source_box'];visible&=(u>=x1)&(u<x2)&(v>=y1)&(v<y2)
 nearest,ix=cKDTree(d['xyz']).query(world[visible])
 return ix[nearest<.015]
def main():
 train=[load('train',id) for id in S['train']]
 val=[load('val',id) for id in S['val']]
 for d in train:d['maps']=visual_maps(d)
 rows=[]
 for split,items in [('train_loo',train),('val',val)]:
  for d in items:
   scores,neighbors=rank(d,train)
   ix=contact_indices(d)
   row=dict(id=d['id'],split=split,visual_neighbors=neighbors,points=len(d['xyz']),evaluable_contacts=len(ix))
   for fraction in (.1,.2,.3):
    n=max(1,math.ceil(fraction*len(d['xyz'])))
    measure=dict(random=n/len(d['xyz']))
    for name,score in [('mean_visual_support',scores[0]),('event_frequency',scores[1])]:
     top=np.argpartition(score,-n)[-n:]
     measure[name+'_contact']=float(np.isin(ix,top).mean()) if len(ix) else None
     teacher=np.argpartition(d['target'],-n)[-n:]
     measure[name+'_visual_overlap']=float(np.isin(teacher,top).mean())
    row['budget_'+str(int(fraction*100))]=measure
   rows.append(row);print(json.dumps(row),flush=True)
 summary={}
 for split in ('train_loo','val'):
  rr=[r for r in rows if r['split']==split and r['evaluable_contacts']]
  summary[split]=dict(total=sum(x['split']==split for x in rows),evaluable=len(rr))
  for b in ('10','20','30'):
   for name in ('mean_visual_support_contact','event_frequency_contact','random','mean_visual_support_visual_overlap','event_frequency_visual_overlap'):
    summary[split][b+'_'+name]=float(np.mean([x['budget_'+b][name] for x in rr])) if rr else None
 out=dict(status='train_visual_retrieval_baseline',retrieval='top five by first-frame frozen DINO similarity; no names or object identities',aggregation='normalized first-frame proposal UV; equal-demo frequency preserves multiple modes; no simulator geometry',reference='old unreviewed simulator contacts, evaluation only',summary=summary,rows=rows)
 target=R/'audits/v3/stage2_crossdemo_frequency_audit.json'
 target.write_text(json.dumps(out,indent=2,ensure_ascii=False))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

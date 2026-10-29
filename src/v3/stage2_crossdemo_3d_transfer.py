"""Train-only RGB-D surface transfer baseline. No names or simulator truth in retrieval/transfer."""
from pathlib import Path
import json,math
import numpy as np
from scipy.spatial import cKDTree
from stage2_crossdemo_frequency import R,S,load,contact_indices,rank,visual_maps
def transfer(query,train):
 candidates=[d for d in train if d['id']!=query['id']]
 sims=np.asarray([query['embedding']@d['embedding'] for d in candidates])
 order=np.argsort(-sims)[:5]
 weights=np.exp((sims[order]-sims[order].max())/.03);weights/=weights.sum()
 qxyz=query['xyz']-query['xyz'].mean(0)
 qfeat=query['dino_feat'];qfeat=qfeat/np.maximum(np.linalg.norm(qfeat,axis=1,keepdims=True),1e-8)
 geom=np.zeros(len(qxyz));fused=np.zeros(len(qxyz))
 for j,w in zip(order,weights):
  d=candidates[j]
  txyz=d['xyz']-d['xyz'].mean(0)
  dist,ix=cKDTree(txyz).query(qxyz,k=min(16,len(txyz)))
  if ix.ndim==1:ix=ix[:,None];dist=dist[:,None]
  spatial=np.exp(-dist/.02)
  target=d['target'][ix]
  geom+=w*np.sum(spatial*target,axis=1)/np.maximum(spatial.sum(1),1e-8)
  tfeat=d['dino_feat'];tfeat=tfeat/np.maximum(np.linalg.norm(tfeat,axis=1,keepdims=True),1e-8)
  cos=np.einsum('nd,nkd->nk',qfeat,tfeat[ix])
  feature=np.exp(np.clip((cos-1)/.08,-30,0))
  combined=spatial*feature
  fused+=w*np.sum(combined*target,axis=1)/np.maximum(combined.sum(1),1e-8)
 return geom,fused,[(candidates[j]['id'],float(sims[j])) for j in order]
def main():
 train=[load('train',id) for id in S['train']]
 val=[load('val',id) for id in S['val']]
 for d in train:d['maps']=visual_maps(d)
 rows=[]
 for split,items in [('train_loo',train),('val',val)]:
  for q in items:
   geom,fused,neighbors=transfer(q,train)
   uvscore,_=rank(q,train)
   contacts=contact_indices(q)
   row=dict(id=q['id'],split=split,points=len(q['xyz']),evaluable_contacts=len(contacts),neighbors=neighbors)
   for fraction in (.1,.2,.3):
    n=max(1,math.ceil(fraction*len(q['xyz'])))
    values=dict(random=n/len(q['xyz']))
    for name,score in [('geometry',geom),('geometry_dino',fused),('uv_frequency',uvscore[1])]:
     top=np.argpartition(score,-n)[-n:]
     values[name+'_contact']=float(np.isin(contacts,top).mean()) if len(contacts) else None
     teacher=np.argpartition(q['target'],-n)[-n:]
     values[name+'_visual_overlap']=float(np.isin(teacher,top).mean())
    row['budget_'+str(int(fraction*100))]=values
   rows.append(row);print(json.dumps(row),flush=True)
 summary={}
 for split in ('train_loo','val'):
  rr=[r for r in rows if r['split']==split and r['evaluable_contacts']]
  summary[split]=dict(total=sum(x['split']==split for x in rows),evaluable=len(rr))
  for budget in ('10','20','30'):
   for key in ('geometry_contact','geometry_dino_contact','uv_frequency_contact','random','geometry_visual_overlap','geometry_dino_visual_overlap'):
    summary[split][budget+'_'+key]=float(np.mean([x['budget_'+budget][key] for x in rr])) if rr else None
 output=dict(status='train_only_surface_transfer_audit',retrieval='top 5 first-frame DINO global cosine, no names',transfer='centered 3D nearest surfaces with optional local DINO weighting; no simulator pose',reference='old unreviewed simulator contact evaluation only',summary=summary,rows=rows)
 target=R/'audits/v3/stage2_crossdemo_3d_transfer_audit.json';target.write_text(json.dumps(output,indent=2,ensure_ascii=False))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

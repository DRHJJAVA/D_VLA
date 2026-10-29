import cProfile,pstats,json,torch
from data import B,Dataset,to_device
from network import JointModel
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15)
ds=Dataset('train');r=ds.rows[0];m=JointModel().cuda().eval();prof=cProfile.Profile();prof.enable()
with torch.no_grad():
 for t in [10,47,80]:
  for oid in ds.frame_objects(r,t)[:2]:
   raw=ds.get(r,t,oid,point_limit=128);b=to_device(raw);m(b)
torch.cuda.synchronize();prof.disable();prof.dump_stats(str(B/'logs/teacher_profile.prof'))
s=pstats.Stats(prof);rows=[]
for (fn,line,name),(cc,nc,tt,ct,callers) in sorted(s.stats.items(),key=lambda x:x[1][3],reverse=True)[:25]:
 rows.append({'module':fn.split('/')[-1],'line':line,'function':name,'calls':nc,'self_s':tt,'total_s':ct})
(B/'manifests/teacher_profile.json').write_text(json.dumps(rows,indent=2))

from pathlib import Path
import sys,json,hashlib,traceback
import numpy as np,torch
B=Path(__file__).resolve().parents[1];R=B.parents[2];O=R/'experiments/region_pose/iter_029'
sys.path.insert(0,str(B/'src'));import train as t
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def met(score,d):
 out=t.metrics(score,d);y=d['target'].cpu().numpy().astype(bool);known=d['known'].cpu().numpy()
 for b in [.01,.02,.05,.1,.2,.3]:
  ix=np.argsort(-score,kind='stable')[:max(1,int(np.ceil(len(score)*b)))];s=str(int(100*b))
  out['all_coverage'+s]=float(y[ix].sum()/max(1,y.sum()));out['all_unknown_fraction'+s]=float((~known[ix]).mean());out['all_known_negative_fraction'+s]=float((known[ix]&~y[ix]).mean())
 return out
def main():
 m=json.loads((B/'manifest.json').read_text());oldbase=json.loads((O/'evaluation/baselines.json').read_text());sources={}
 locks={}
 for n in ['014','015','027','028']:locks.update(json.loads((B.parent/f'iter_{n}/completed_locked.json').read_text()))
 files=[O/'evaluation/baselines.json']
 for sp,it in [('train','027'),('val','028')]:
  p=B.parent/f'iter_{it}/summary.json';sources[sp]=json.loads(p.read_text())['rows'];files.append(p)
 tasks=[]
 for r in m['rows']:
  s=next(x for x in sources[r['split']] if x['event']==r['event'] and x['query']==r['query'])
  raw=R/s['source_file'];parent=Path(r['target_source']);files += [raw,parent];tasks.append((r,raw,parent))
 il={}
 for p in files:
  rel=str(p.relative_to(R));h=sha(p)
  if p!=O/'evaluation/baselines.json':assert locks[rel]==h,rel
  il[rel]=h
 dump(B/'baseline_input_locked.json',il)
 dump(B/'baseline_source_locked.json',{str(p.relative_to(R)):sha(p) for p in [B/'src/audit_baselines.py',B/'几何基线索引核查031_1.md']})
 results=[];baselines=[]
 for r,raw,parent in tasks:
  a=np.load(raw);z=np.load(parent);d=np.load(B/'samples'/f"{r['key']}.npz");ix=d['causal_fps_indices'];oldix=z['selected_index']
  assert np.array_equal(a['query_pixel'],z['query_pixel']) and np.array_equal(a['query_xyz'],z['query_xyz'])
  assert np.array_equal(a['query_pixel'][ix],d['pixel']) and np.array_equal(a['query_xyz'][ix].astype(np.float32),d['xyz_world'])
  assert np.array_equal(np.sort(ix),np.sort(oldix))
  td={k:torch.as_tensor(d[k]) for k in ['known','target']}
  original=met(a['early_geometry_score'][oldix],td);corrected=met(a['early_geometry_score'][ix],td)
  old=next(x for x in oldbase if x['key']==r['key'])
  assert all(abs(original[k]-old['current_geometry'][k])<1e-12 for k in original)
  center=met(-np.linalg.norm(d['xyz'],axis=1),td)
  assert all(abs(center[k]-old['center'][k])<1e-12 for k in center)
  new=dict(old);new['current_geometry']=corrected;baselines.append(new)
  results.append({'key':r['key'],'split':r['split'],'episode':r['episode'],'offset':r['offset'],'points':len(ix),'same_index_order':bool(np.array_equal(ix,oldix)),'geometry_score_max_difference':float(np.max(np.abs(a['early_geometry_score'][ix]-a['early_geometry_score'][oldix]))),'old_ap':original['ap'],'corrected_ap':corrected['ap'],'sample_input_unchanged':True,'center_exact':True})
 report={'rows':results,'changed_order_queries':sum(not x['same_index_order'] for x in results),'split_means':{sp:{'queries':sum(x['split']==sp for x in results),'old_ap':float(np.mean([x['old_ap'] for x in results if x['split']==sp])),'corrected_ap':float(np.mean([x['corrected_ap'] for x in results if x['split']==sp])),'changed_order':sum(x['split']==sp and not x['same_index_order'] for x in results)} for sp in ['train','val']},'episode_means':{ep:{'old_ap':float(np.mean([x['old_ap'] for x in results if x['episode']==ep])),'corrected_ap':float(np.mean([x['corrected_ap'] for x in results if x['episode']==ep]))} for ep in sorted({x['episode'] for x in results})},'no_target_or_model_change':True,'old029_and030_outputs_preserved':True,'random_reused_unchanged':True,'center_recomputed_equal':True}
 dump(B/'evaluation/baselines.json',baselines);dump(B/'audits/baseline_index_correction.json',report)
 assert all(sha(R/k)==v for k,v in il.items())
 print(json.dumps({'baseline_audit_complete':True,'summary':report['split_means']}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  (B/'logs/baseline_error_remote_only.txt').write_text(traceback.format_exc());print(json.dumps({'type':type(e).__name__,'message':str(e)[:200]}));raise SystemExit(1)

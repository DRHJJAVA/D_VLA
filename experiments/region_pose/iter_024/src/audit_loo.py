from pathlib import Path
import json,hashlib,sys,time
import numpy as np,cv2
import audit_regions as reg
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose';O=B/'iter_024'
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):
 a=np.asarray(a)
 if not a.size:return {'n':0,'median_mm':None,'p95_mm':None,'max_mm':None,'fraction_le2mm':None}
 return {'n':int(a.size),'median_mm':float(np.median(a)*1000),'p95_mm':float(np.percentile(a,95)*1000),'max_mm':float(a.max()*1000),'fraction_le2mm':float((a<=.002).mean())}
def expected_frames(c):
 selected=list(c['selected_held_frames']);br={int(x['frame']):x for x in c['bridge_results']};out=[selected[0]]
 assert br[selected[0]]['valid']
 cmp={int(x['frames'][1]):x for x in c['time_comparisons']}
 for f in selected[1:]:
  if not br[f]['valid'] or not cmp.get(f,{}).get('valid',False):break
  out.append(f)
 return out
def main(split):
 source=B/('iter_022' if split=='train' else 'iter_023');out=O/(split+'_loo')
 assert not (out/'summary.json').exists()
 cs=[c for c in json.loads((source/split/'summary.json').read_text())['cases'] if c['valid']]
 region=json.loads((O/split/'summary.json').read_text())
 # Enumerate the region source records and preserve all current-frame identities.
 records=[]
 for c in cs:
  ident=c['id'];stem=c['event']+'_query'+str(c['query']);z=np.load(O/split/(stem+'.npz'));cr=np.load(source/split/'correspondences'/(stem+'.npz'));seed=np.load(source/'seeds'/(ident+'.npz'))
  frames=z['selected_frames'].astype(int).tolist();assert frames==expected_frames(c)
  assert np.array_equal(z['query_xyz'],cr['query_foreground_xyz'])
  query=z['query_xyz'];fr=[]
  for t in frames:
   Q=z['query_to_current_hand_'+str(t)];H=z['hand_to_world_'+str(t)]
   br=next(x for x in c['bridge_results'] if int(x['frame'])==t)
   assert np.array_equal(Q,np.asarray(br['query_world_to_anchor_hand']))
   assert np.array_equal(H,cr['hand_to_world_'+str(t)])
   inl=cr['inlier_'+str(t)].astype(bool);ids=cr['track_id_'+str(t)][inl].astype(int)
   a=cr['query_xyz_'+str(t)][inl];b=cr['bridge_xyz_'+str(t)][inl];uv=cr['query_uv_'+str(t)][inl]
   src=z['source_xyz'][z['frame']==t];mapped=z['mapped_query_xyz'][z['frame']==t]
   W=H@Q;Wi=np.linalg.inv(W);base=reg.apply(src,Wi)
   assert np.allclose(base,mapped,atol=1e-9)
   assert np.allclose(reg.apply(base,W),src,atol=1e-9)
   fits=[];deleted=[];rejected=[];rem_scale=[]
   for j in range(len(a)):
    aa=np.delete(a,j,axis=0);bb=np.delete(b,j,axis=0)
    if len(aa)<7:rejected.append({'track_id':int(ids[j]),'reason':'remaining_points_lt7'});continue
    scale=np.linalg.svd(aa-aa.mean(0),compute_uv=False)/np.sqrt(len(aa))
    if scale[1]<.003:rejected.append({'track_id':int(ids[j]),'reason':'second_axis_lt3mm','scale_mm':float(scale[1]*1000)});continue
    fits.append(np.linalg.inv(H)@reg.rigid(aa,bb));deleted.append(int(ids[j]));rem_scale.append(scale)
   eligible=len(fits)>=7;query_d=[];region_d=[]
   for Qalt in fits:
    query_d.append(np.linalg.norm(reg.apply(query,Qalt)-reg.apply(query,Q),axis=1))
    region_d.append(np.linalg.norm(reg.apply(src,np.linalg.inv(H@Qalt))-base,axis=1))
   qmax=np.max(query_d,axis=0) if fits else np.full(len(query),np.nan)
   rmax=np.max(region_d,axis=0) if fits else np.full(len(src),np.nan)
   origin=seed['source_is_grid'][ids].astype(bool);deleted=np.asarray(deleted,int);groups=seed['source_is_grid'][deleted].astype(bool)
   group_stats={}
   for name,isgrid in [('original',False),('grid',True)]:
    select=np.where(groups==isgrid)[0]
    group_stats[name]={'valid_loo_fits':int(len(select)),'query_max_shift':stats(np.max(np.asarray(query_d)[select],axis=0)) if len(select) else stats([]),'region_max_shift':stats(np.max(np.asarray(region_d)[select],axis=0)) if len(select) else stats([])}
   np.savez_compressed(out/(stem+'_frame'+str(t)+'.npz'),nominal_Q=Q,hand_to_world=H,loo_Q=np.asarray(fits).reshape(-1,4,4),removed_track_id=deleted,removed_is_grid=groups,inlier_track_id=ids,inlier_is_grid=origin,query_max_shift_m=qmax,region_max_shift_m=rmax,region_source_xyz=src,region_mapped_query_xyz=base,remaining_principal_scales=np.asarray(rem_scale).reshape(-1,3))
   scale=np.linalg.svd(a-a.mean(0),compute_uv=False)/np.sqrt(len(a))
   frame={'frame':t,'inliers':len(a),'original_inliers':int((~origin).sum()),'grid_inliers':int(origin.sum()),'query_uv_hull_px2':float(cv2.contourArea(cv2.convexHull(uv.astype(np.float32)))) if len(uv)>=3 else 0.,'query_uv_grid8_cells':len(set(map(tuple,(uv//8).astype(int)))),'query_xyz_principal_scales_mm':(scale*1000).tolist(),'valid_loo_fits':len(fits),'eligible_min7_loo':eligible,'rejected':rejected,'query_max_shift':stats(qmax) if fits else stats([]),'region_max_shift':stats(rmax) if fits else stats([]),'source_type_deletion':group_stats,'raw_region_points':len(src),'nominal_roundtrip_verified':True,'sensitivity_not_physical_error_truth':True}
   fr.append(frame)
  rj=json.loads((O/split/(stem+'.json')).read_text())
  row={'episode':ident,'event':c['event'],'query':c['query'],'selected_frames':frames,'omitted_later_endpoints':[x for x in c['selected_held_frames'] if x not in frames],'frames':fr,'all_frames_have_min7_loo':all(x['eligible_min7_loo'] for x in fr),'raw_points':rj['raw_points'],'supported_1mm_voxels':rj['supported_1mm_voxels'],'query_surface_points':len(query),'weak_minimum_support':any(x['inliers']<=8 for x in fr),'training_labels_created':False}
  records.append(row);save(out/(stem+'.json'),row)
 summary={'split':split,'events':len(records),'cases':records,'not_grasp_success_rate':True,'region_is_not_physical_contact_truth':True,'targets_created':False,'training_performed':False,'time':time.time()}
 save(out/'summary.json',summary)
 save(out/'locked.json',{str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file() and p.name!='locked.json'})
 print(json.dumps({'split':split,'loo_events':len(records),'all_fit_eligible_events':sum(r['all_frames_have_min7_loo'] for r in records)}))
if __name__=='__main__':main(sys.argv[1])

from pathlib import Path
import json,datetime
import h5py,numpy as np
R=Path("/mnt/nvme1/drh/D_VLA")
rows=[]
for p in sorted((R/"data/v2/stage1_rgbd").glob("*/*.h5")):
 with h5py.File(p,"r") as f:
  n=int(f.attrs.get("frames",f.attrs.get("num_frames")))
  cams=[k for k in f if isinstance(f[k],h5py.Group)]
  src=Path(str(f.attrs.get("source",f.attrs.get("source_hdf5"))));src=src if src.is_absolute() else R/src;episode=str(f.attrs["episode"])
  if "frame_indices" in f: assert np.array_equal(f["frame_indices"][:],np.arange(n))
  with h5py.File(src,"r") as sf:
   original_n=len(sf["data"][episode]["states"])
  assert n==original_n,(p,n,original_n)
  covered=[i for start in range(0,n,16) for i in range(start,min(start+16,n))]
  assert covered==list(range(n))
  cr={}
  for cam in cams:
   g=f[cam];rgb=g["rgb"];d=g["depth_m"]
   assert len(rgb)==len(d)==len(g["K"])==len(g["camera_to_world"])==n
   assert rgb.shape[1:3]==d.shape[1:3] and rgb.shape[-1]==3
   k=g["K"][:];t=g["camera_to_world"][:]
   assert np.isfinite(k).all() and np.isfinite(t).all()
   assert (k[:,0,0]>0).all() and (k[:,1,1]>0).all()
   assert np.allclose(t[:,3,:],np.array([0,0,0,1]),atol=1e-5)
   finite=True;positive=0;total=0
   for start in range(0,n,16):
    dep=d[start:start+16]
    finite=finite and bool(np.isfinite(dep).all())
    positive+=int((dep>0).sum());total+=dep.size
   assert finite
   cr[cam]={"rgb_shape":list(rgb.shape),"depth_shape":list(d.shape),"all_depth_frames_finite":finite,"positive_depth_fraction":positive/total,"all_calibrations_finite":True}
  rows.append({"object":p.parent.name,"episode":episode,"frames":n,"source_frames":original_n,"full_sequence_chunk_coverage":True,"cameras":cr})
result={"generated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"status":"input_contract_passed_not_training","episodes":len(rows),"objects":sorted(set(r["object"] for r in rows)),"synchronized_time_steps":sum(r["frames"] for r in rows),"camera_frames":sum(r["frames"]*len(r["cameras"]) for r in rows),"all_frames_included":True,"all_depth_frames_checked":True,"rgb_semantics_or_visual_alignment_audited_by_this_script":False,"actions_or_region_labels_read":False,"model_trained":False,"rows":rows}
out=R/"audits/v3/full_sequence_input_check.json";out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!="rows"},ensure_ascii=False))

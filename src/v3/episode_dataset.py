"""Read every supplied RGB-D frame; no action-phase or interaction-count assumptions."""
from pathlib import Path
import json,h5py,numpy as np
CAMERAS=("agentview","robot0_eye_in_hand")
FIELDS=("rgb","depth_m","K","camera_to_world")
def read_manifest(path):
 rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
 ids=[r["id"] for r in rows]
 if len(ids)!=len(set(ids)):raise ValueError("Repeated episode identifier")
 return rows
def iter_episode_chunks(entry,chunk_size=16):
 if chunk_size<1:raise ValueError("chunk_size must be positive")
 with h5py.File(entry["rgbd_path"],"r") as f:
  n=int(entry["frames"])
  timestamps=np.asarray(entry["timestamps_seconds"],dtype=np.float64)
  if len(timestamps)!=n:raise ValueError("Timestamp count differs from manifest")
  if "_generated_tail" in f or int(f.attrs.get("synthetic_tail",0))>0:raise ValueError("Synthetic extension forbidden")
  for cam in CAMERAS:
   for key in FIELDS:
    if len(f[cam+"/"+key])!=n:raise ValueError("Manifest frame count differs from RGB-D")
  for start in range(0,n,chunk_size):
   end=min(start+chunk_size,n)
   yield {"episode_id":entry["id"],"frame_indices":np.arange(start,end,dtype=np.int64),"timestamps":timestamps[start:end],"cameras":{cam:{key:f[cam+"/"+key][start:end] for key in FIELDS} for cam in CAMERAS},"first_chunk":start==0,"last_chunk":end==n}
def audit_loader(manifest,output):
 rows=read_manifest(manifest);reports=[]
 for row in rows:
  expected=0;counts={cam:0 for cam in CAMERAS};last_time=None;finite=True
  for chunk in iter_episode_chunks(row):
   ii=chunk["frame_indices"];tt=chunk["timestamps"]
   assert np.array_equal(ii,np.arange(expected,expected+len(ii)))
   if last_time is not None:assert tt[0]>last_time
   assert np.all(np.diff(tt)>0);last_time=tt[-1];expected+=len(ii)
   assert set(chunk["cameras"])==set(CAMERAS)
   for cam,g in chunk["cameras"].items():
    assert set(g)==set(FIELDS)
    assert all(len(v)==len(ii) for v in g.values())
    assert np.isfinite(g["depth_m"]).all() and (g["depth_m"]>0).all()
    counts[cam]+=len(ii)
  assert expected==row["frames"] and all(v==expected for v in counts.values())
  reports.append({"id":row["id"],"frames":expected,"camera_frames":counts,"chronological_complete":True})
 result={"status":"passed","episodes":len(rows),"synchronized_time_steps":sum(r["frames"] for r in reports),"camera_frames":sum(sum(r["camera_frames"].values()) for r in reports),"network_field_allowlist":list(FIELDS),"actions_or_region_labels_loaded":False,"stage":"data_only_no_training","rows":reports}
 Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2))
 return result
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 r=audit_loader(a.manifest,a.output);print(json.dumps({k:v for k,v in r.items() if k!="rows"}))

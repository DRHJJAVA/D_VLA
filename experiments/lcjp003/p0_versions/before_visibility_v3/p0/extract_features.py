import os,sys,json,time,pathlib,traceback,hashlib,datetime
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
B=pathlib.Path("/mnt/nvme1/drh/D_VLA"); E=B/"experiments/lcjp003"; P=E/"p0"
sys.path.insert(0,str(E/"src"))
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for x in iter(lambda:f.read(4*1024*1024),b""):h.update(x)
 return h.hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds")
def write(p,x):
 tmp=p.with_suffix(".tmp.json");tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def stats(a,valid=None):
 if valid is not None:a=a[valid]
 return {"shape":list(a.shape),"finite":bool(np.isfinite(a).all()),"min":float(a.min()) if a.size else None,"max":float(a.max()) if a.size else None,"mean":float(a.mean()) if a.size else None,"std":float(a.std()) if a.size else None}
t=time.time();results=[]
try:
 import numpy as np,torch,fcntl
 lock=open(P/"features_extract.lock","a")
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 from features import FrozenSmolFeatures
 audit_path=P/"features_input_manifest.json";audit_sha=sha(audit_path);audit=json.loads(audit_path.read_text());source_sha=sha(E/"src/features.py")
 assert len(audit["samples"])==10
 assert audit["heldout_or_dev_content_read"] is False and audit["raw_readonly"] is True
 out=P/"features";out.mkdir(exist_ok=True)
 write(P/"features_state.json",{"status":"loading_frozen_backbone","pid":os.getpid(),"created_beijing":stamp(),"data_audit_sha256":audit_sha})
 enc=FrozenSmolFeatures(str(E/"weights/smolvlm2"),"cuda")
 assert len(enc.text.layers)==16 and not enc.model.training
 assert all(not x.requires_grad for x in enc.model.parameters())
 for item in audit["samples"]:
  st=time.time();src=B/item["path"];input_sha=sha(src)
  assert input_sha==item["sha256"]
  with np.load(src,allow_pickle=False) as z:
   rgb=z["rgb"];uv=z["projected_uv"];vv=z["view_visible"];pv=z["point_valid"];hv=z["history_valid"];fi=z["frame_indices"]
   assert np.all(fi[hv]<=item["current_frame"]) and item["no_future_frame_read"]
   semantic=np.asarray(enc.point_features(rgb,uv,vv),dtype=np.float32)
  text,text_valid=enc.text_features(item["instruction"])
  text=np.asarray(text,dtype=np.float32);text_valid=np.asarray(text_valid,dtype=bool)
  assert semantic.shape==(16,2048,768) and text.ndim==2 and text.shape[-1]==960 and text_valid.shape==(len(text),)
  assert np.isfinite(semantic).all() and np.isfinite(text).all()
  assert sha(src)==input_sha and sha(audit_path)==audit_sha and sha(E/"src/features.py")==source_sha
  assert not any(x.grad is not None or x.requires_grad for x in enc.model.parameters())
  dst=out/(src.stem+".npz");tmp=out/(src.stem+".tmp.npz")
  np.savez_compressed(tmp,semantic=semantic,text_features=text,text_valid=text_valid,sample_sha256=np.asarray(input_sha));tmp.replace(dst)
  known=vv.any(-1)&pv&hv[:,None]
  invalid=~vv.any(-1)
  results.append({"episode_id":item["episode_id"],"current_frame":item["current_frame"],"case":item["case"],"input_path":str(src.relative_to(B)),"input_sha256":input_sha,"feature_path":str(dst.relative_to(B)),"feature_sha256":sha(dst),"elapsed_seconds":time.time()-st,"semantic":{"shape":list(semantic.shape),"dtype":str(semantic.dtype),"all_finite":True,"visible_valid":stats(semantic,known),"no_visible_view_count":int(invalid.sum()),"no_visible_view_abs_max":float(np.abs(semantic[invalid]).max()) if invalid.any() else 0.0},"text":{"shape":list(text.shape),"valid_count":int(text_valid.sum()),"all_finite":True},"causal_frame_max":int(fi[hv].max()),"current_frame_bound":item["current_frame"],"history_valid_count":int(hv.sum())})
  write(P/"features_state.json",{"status":"extracting","pid":os.getpid(),"updated_beijing":stamp(),"complete":len(results),"total":10,"elapsed_seconds":time.time()-t})
 fa=enc.finish_audit()
 assert fa["unchanged"] and not fa["any_gradient"] and fa["requires_grad_count"]==0
 assert sha(audit_path)==audit_sha and sha(E/"src/features.py")==source_sha
 result={"status":"completed","created_beijing":stamp(),"elapsed_seconds":time.time()-t,"data_audit_sha256":audit_sha,"input_manifest":"experiments/lcjp003/p0/features_input_manifest.json","features_source_sha256":source_sha,"samples":results,"frozen_backbone":fa,"cache_schema":{"semantic":"float32[T=16,N=2048,D=768]","text_features":"float32[L,960]","text_valid":"bool[L]","sample_sha256":"unicode string scalar"},"all_features_finite":True,"all_input_hashes_match":True,"all_feature_inputs_causal":True,"heldout_or_dev_content_read":False,"raw_readonly":True,"purpose":"P0 real frozen features; no contact labels; no learning performed","projection_calibration_gate":"under_separate_diagnosis_not_passed_by_feature_finiteness","scope":"mechanical_network_gradient_check_with_current_locked_sample_hashes"}
 write(P/"features_audit.json",result)
 write(P/"features_state.json",{"status":"completed","pid":os.getpid(),"updated_beijing":stamp(),"complete":10,"total":10,"elapsed_seconds":time.time()-t,"audit_sha256":sha(P/"features_audit.json")})
except BaseException as e:
 write(P/"features_failure.json",{"status":"failed","created_beijing":stamp(),"error_type":type(e).__name__,"frames":[{"file":pathlib.Path(x.filename).name,"line":x.lineno,"function":x.name} for x in traceback.extract_tb(e.__traceback__)],"completed":len(results),"elapsed_seconds":time.time()-t})
 write(P/"features_state.json",{"status":"failed","pid":os.getpid(),"completed":len(results),"elapsed_seconds":time.time()-t})

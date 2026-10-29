import os,sys,json,time,pathlib,traceback,hashlib
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("MKL_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
B=pathlib.Path("/mnt/nvme1/drh/D_VLA")
P=B/"experiments/lcjp003/p0"
sys.path.insert(0,str(B/"experiments/lcjp003/src"))
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for x in iter(lambda:f.read(4*1024*1024),b""):h.update(x)
 return h.hexdigest()
def save(x): (P/"feature_probe_state.json").write_text(json.dumps(x,ensure_ascii=False,indent=2))
t=time.time()
try:
 import numpy as np,torch
 from features import FrozenSmolFeatures
 audit=json.loads((P/"data_audit.json").read_text())
 item=audit["samples"][0]; src=B/item["path"]; before=sha(src)
 assert before==item["sha256"]
 save({"status":"loading_backbone","pid":os.getpid(),"sample_sha256":before})
 enc=FrozenSmolFeatures(str(B/"experiments/lcjp003/weights/smolvlm2"),"cuda")
 save({"status":"extracting_one_window","pid":os.getpid(),"sample_sha256":before,"frozen":all(not p.requires_grad for p in enc.model.parameters()),"text_layers":len(enc.text.layers)})
 with np.load(src,allow_pickle=False) as z:
  semantic=enc.point_features(z["rgb"],z["projected_uv"],z["view_visible"])
 text,text_valid=enc.text_features(item["instruction"])
 fa=enc.finish_audit()
 assert sha(src)==before
 semantic=np.asarray(semantic);text=np.asarray(text);text_valid=np.asarray(text_valid)
 assert semantic.shape==(16,2048,768)
 assert text.ndim==2 and text.shape[-1]==960
 assert text_valid.shape==(text.shape[0],)
 assert np.isfinite(semantic).all() and np.isfinite(text).all()
 assert not any(p.requires_grad or p.grad is not None for p in enc.model.parameters())
 d=P/"features_probe"; d.mkdir(exist_ok=True)
 dst=d/(src.stem+".npz")
 np.savez_compressed(dst,semantic=semantic,text_features=text,text_valid=text_valid.astype(bool))
 save({"status":"complete_one_window_provisional","pid":os.getpid(),"elapsed_seconds":time.time()-t,"sample_path":str(src.relative_to(B)),"sample_sha256":before,"feature_path":str(dst.relative_to(B)),"feature_sha256":sha(dst),"semantic":{"shape":list(semantic.shape),"dtype":str(semantic.dtype),"min":float(semantic.min()),"max":float(semantic.max()),"mean":float(semantic.mean()),"std":float(semantic.std()),"finite":bool(np.isfinite(semantic).all())},"text":{"shape":list(text.shape),"dtype":str(text.dtype),"finite":bool(np.isfinite(text).all()),"valid_count":int(text_valid.sum())},"frozen_audit":fa,"source_sha256":sha(B/"experiments/lcjp003/src/features.py"),"not_final_cache":True})
except BaseException as e:
 save({"status":"failure","error_type":type(e).__name__,"frames":[{"file":pathlib.Path(x.filename).name,"line":x.lineno,"function":x.name} for x in traceback.extract_tb(e.__traceback__)],"elapsed_seconds":time.time()-t})

from pathlib import Path
import os,json,importlib.util,traceback
os.environ["MUJOCO_GL"]="egl"
import numpy as np
from scipy.stats import norm
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path("/mnt/nvme1/drh/D_VLA");O=R/"experiments/contact_preference/stage3_pilot_v2"
sp=importlib.util.spec_from_file_location("b",O/"src/build_supports.py");b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
def decomp(d,s,model):
 lc=norm.logpdf(d,0,s);ls=b.logsep(d,s,model["mu"],model["family"]);p=np.exp(lc-np.logaddexp(lc,ls))
 return np.maximum(2*p-1,0),np.maximum(1-2*p,0)
def main():
 (O/"points_evidence").mkdir(exist_ok=True);S=json.loads((O/"summary.json").read_text());models=json.loads((O/"fitted_models.json").read_text())
 out=[];tests=[];source_derived={}
 for ci,ca in enumerate(S["cases"]):
  feat=np.load(O/"points"/("event_features_c"+str(ci)+".npz"));ms=[x for x in models if x["target_episode"]==ci]
  ep=[decomp(feat["mean_signed_gap"],feat[x["error_model"]],x) for x in ms]
  epP=np.stack([x[0] for x in ep]).min(0);epN=np.stack([x[1] for x in ep]).min(0);rows=[]
  for row in ca["rows"]:
   z=np.load(O/row["data"]);inst=[decomp(z["signed_distance_m"],z["sigma_base_m"] if x["error_model"]=="base" else z["sigma_proxy_m"],x) for x in ms]
   P=np.minimum(epP,np.stack([x[0] for x in inst]).min(0));N=np.minimum(epN,np.stack([x[1] for x in inst]).min(0))
   P[~z["opposing_normals"]]=0
   scope=z["excluded_mounting_nearest"];deep=z["deep_penetration"];P[scope|deep]=0;N[scope|deep]=0
   U=1-P-N;mass=np.column_stack([P,N,U]);cl=mass.argmax(1);cl[(np.isclose(mass,mass.max(1)[:,None],atol=1e-12,rtol=0).sum(1)>1)]=2
   cl[scope]=3;cl[deep&~scope]=4
   data={k:z[k] for k in z.files if k not in ['positive','separation','unknown','display_class','support_class']}
   data.update(positive=P,separation=N,unknown=U,display_class=cl,mixture_attempt_positive=z['positive'],mixture_attempt_separation=z['separation'])
   fn=Path(row["data"]).name;np.savez_compressed(O/"points_evidence"/fn,**data)
   assert np.all(P>=0) and np.all(N>=0) and np.all(U>=-1e-12) and np.allclose(mass.sum(1),1)
   assert np.all(P[scope|deep|~z["opposing_normals"]]==0)
   r=dict(row);r.update(data="points_evidence/"+fn,count={str(i):int((cl==i).sum()) for i in range(5)},
    positive_max=float(P.max()),positive_mean=float(P.mean()),green_visible=int(((cl==0)&z["currently_depth_consistent"]).sum()),
    green_occluded=int(((cl==0)&~z["currently_depth_consistent"]).sum()),
    non_opposing_positive_count=int(((cl==0)&~z["opposing_normals"]).sum()),
    mounting_positive_count=int(((cl==0)&scope).sum()),
    green_gap_mm_range=[float(z["signed_distance_m"][cl==0].min()*1000),float(z["signed_distance_m"][cl==0].max()*1000)] if (cl==0).any() else None)
   rows.append(r)
  out.append(dict(**{k:v for k,v in ca.items() if k!="rows"},rows=rows))
  sig=float(np.median(feat["base"]));d=np.array([0,1,3,10])*sig
  ss=[decomp(d,np.full(4,sig),x) for x in ms];pp=np.stack([x[0] for x in ss]).min(0);nn=np.stack([x[1] for x in ss]).min(0)
  # Noise made noninformative compared with the fitted gap scale should lose support.
  huge=max(x["mu"] for x in ms)*10
  uncertainty=[decomp(np.array([0.]),np.array([huge]),x) for x in ms]
  noisy=min(x[0][0] for x in uncertainty)
  assert pp[0]>pp[-1] and pp[-1]==0 and noisy<pp[0]
  tests.append(dict(episode=ci,sigma_mm=sig*1000,gap_in_sigma=[0,1,3,10],positive=pp.tolist(),separation=nn.tolist(),
   unknown=(1-pp-nn).tolist(),very_large_noise_positive=float(noisy),note="1sigma gap may still support contact compatibility, cannot prove exact physical contact"))
 b.save(O/"evidence_summary.json",dict(time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),cases=out,
  synthetic_tests=tests,scoring_network_trained=False,support_is_calibrated_probability=False,
  implementation="likelihood_excess_support_with_overlap_unknown; mounting_excluded"))
 b.save(O/"stage_state.json",dict(stage="evidence_computed_pending_audit",time=b.now()))
 print(json.dumps(dict(cases=[dict(key=c["key"],rows=[dict(frame=r["frame"],count=r["count"],max=r["positive_max"]) for r in c["rows"]]) for c in out])))
if __name__=="__main__":
 try:main()
 except Exception as e:
  (O/"evidence_failure.log").write_text(traceback.format_exc())
  print(json.dumps(dict(error=type(e).__name__,message=str(e))));raise SystemExit(1)

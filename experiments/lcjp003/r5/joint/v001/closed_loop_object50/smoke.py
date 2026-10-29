import os
os.environ.setdefault("MUJOCO_GL","egl")
os.environ.setdefault("PYOPENGL_PLATFORM","egl")
import sys,pathlib,json,time,traceback
RUN=pathlib.Path("/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r5/joint/v001")
sys.path.insert(0,str(RUN/"src"))
import torch,numpy as np
from io_runtime import setup,seed,build,move,atomic,OLD
from adapter import Front,render_module
from r5_model import cloud_summary
def main():
 setup();seed(20261029);model=build();model.eval();front=Front();renderer=render_module()
 es=json.loads((RUN/"episodes.json").read_text());e=es[0];eid=e["id"]
 words=torch.load(RUN/"closed_loop_object50/task_words.pt",weights_only=False)[e["task"]]
 render=renderer.LiveRender(eid,RUN/"closed_loop_object50/engineering_smoke",capacity=4)
 d=move(front.inputs(render.out,0,words,[]));cached=move(torch.load(OLD/"cache"/eid/"0.pt",weights_only=False)["inputs"])
 differences={}
 for k in ["features","rgb","depth","robot","K","T","ee","rotation","fingers","h","words"]:
  differences[k]=float((d[k].float()-cached[k].float()).abs().max())
  assert differences[k]==0,"initial_input_"+k
 with torch.no_grad():
  f=model.front_frame(d);mask,_=model.chosen(f);now=cloud_summary(d,mask.sigmoid())
  a=model.encode(d,history_frames=[d]*16,front_override=f)
  b=model.encode(d,front_override=f,causal_summaries=now[None].repeat(15,1))
  prefix_difference=float((a["prefix"]-b["prefix"]).abs().max())
  assert prefix_difference<=1e-6,"cached_causal_history_prefix"
  noise=torch.randn(1,16,6,device="cuda");ma,ga=model.generate(a,noise);mb,gb=model.generate(b,noise)
  delta=max(float((ma-mb).abs().max()),float((ga-gb).abs().max()));assert delta<=1e-6,"cached_history_actions"
  norm=json.loads((RUN/"normalization.json").read_text());raw=mb[0,0].cpu().numpy()*np.array(norm["std"])+np.array(norm["mean"])
  action=np.r_[raw,float(gb[0,0].argmax())*2-1].astype(np.float32)
 _,done=render.step(np.clip(action,-1,1),1)
 d1=front.inputs(render.out,1,words,[np.clip(action,-1,1)])
 assert all(torch.isfinite(v).all() for v in d1.values() if torch.is_tensor(v))
 render.close()
 atomic("closed_loop_object50/ready.json",{"passed":True,"initial_input_max_differences":differences,"cached_history_prefix_difference":prefix_difference,"cached_history_action_difference":delta,"real_predict_step_observe_cycles":1,"official_test_trials_run":0,"trained_weight_used":False,"video_generated":False,"time":time.time()})
if __name__=="__main__":
 try:main()
 except BaseException as ex:
  atomic("closed_loop_object50/smoke_failure.json",{"type":type(ex).__name__,"message":str(ex)[:200],"frames":[{"file":f.filename,"line":f.lineno} for f in traceback.extract_tb(ex.__traceback__)],"time":time.time()})
  raise SystemExit(1)

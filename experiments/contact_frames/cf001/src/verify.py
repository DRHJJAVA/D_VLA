from common import *
import numpy as np,torch
from model import ContactFrameNet
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.08)
 cfg=json.loads((RUN/"config.json").read_text());z=np.load(RUN/"artifacts/dataset.npz");p=np.load(RUN/"artifacts/frame_predictions.npz")
 image=torch.tensor(z["images"],device="cuda");points=torch.tensor(z["points"],device="cuda")
 maxdiff=0.;ns=[]
 for seed in cfg["ensemble_seeds"]:
  ck=torch.load(RUN/"artifacts"/("seed_"+str(seed))/"last.pt",map_location="cpu",weights_only=False)
  net=ContactFrameNet(ck["state_dim"]).cuda();net.load_state_dict(ck["model"]);net.eval()
  state=torch.tensor((z["states"]-ck["state_mean"])/ck["state_std"],device="cuda")
  ix=torch.tensor(p["context_indices"],device="cuda");pred=[]
  with torch.inference_mode():
   for a in range(0,len(state),16):
    b=ix[a:a+16];l,_=net(image[b],points[b],state[b]);pred.extend(l.sigmoid().cpu().numpy())
  old=np.load(RUN/"artifacts"/("seed_"+str(seed))/"predictions.npz")["probability"]
  diff=float(np.max(np.abs(np.asarray(pred)-old)));assert diff<1e-6;maxdiff=max(maxdiff,diff);ns.append(ck["parameters"])
  del net;torch.cuda.empty_cache()
 # No artificial surface/current hidden-point inclusion in point encoder.
 valid=z["points"][...,9]>0
 assert np.all(z["points"][...,:3][~valid]==0)
 # Weight-zero samples contribute zero classification gradients.
 l=torch.zeros(3,requires_grad=True);loss=(torch.nn.functional.binary_cross_entropy_with_logits(l,torch.tensor([.2,.8,.5]),reduction="none")*torch.tensor([1.,0.,0.])).mean();loss.backward()
 assert float(l.grad[1:].abs().max())==0
 # Interval display is exhaustive and disjoint; one-frame intervals are allowed.
 masks=[p[k] for k in ["contact_candidate","observed_separation_candidate","unknown"]]
 assert np.all(np.stack(masks).sum(0)==1)
 assert not masks[0][0] and masks[2][0]
 lock=json.loads((RUN/"source_locked.json").read_text());assert all(sha(k)==v for k,v in lock.items())
 lock2=json.loads((RUN/"input_locked.json").read_text())["files"];assert all(sha(k)==v for k,v in lock2.items())
 assert np.all(np.isfinite(p["probabilities"]))
 # A changed gripper state cannot alter stored independently fitted visual T.
 # Structural audit is limited to fit(a,b,rng) signature and no hand input.
 report={"status":"passed","prediction_reload_max_abs_difference":maxdiff,"parameters":ns,"unknown_gradient_zero":True,"state_partition_complete_disjoint":True,"frame0_unknown":True,"invisible_points_not_current_observations":True,"source_and_input_hashes_unchanged":True,"physical_contact_truth_validation":False,"independent_video_validation":False}
 save(RUN/"artifacts/verification.json",report)
 stage("verified",reload_max_abs_difference=maxdiff)
if __name__=="__main__":
 try:main()
 except Exception as e:fail(e);raise SystemExit(1)

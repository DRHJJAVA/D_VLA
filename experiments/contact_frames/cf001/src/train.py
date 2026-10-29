from common import *
import numpy as np,torch,random,csv
from torch import nn
from model import ContactFrameNet
def intervals(mask,times):
 ids=np.flatnonzero(mask);out=[]
 for group in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
  if len(group):out.append({"start_frame":int(group[0]),"end_frame":int(group[-1]),"frame_count":len(group),"start_seconds":float(times[group[0]]),"end_seconds":float(times[group[-1]])})
 return out
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.10)
 cfg=json.loads((RUN/"config.json").read_text());data=np.load(RUN/"artifacts/dataset.npz")
 image=torch.tensor(data["images"],device="cuda");point=torch.tensor(data["points"],device="cuda")
 trainmask=data["train"];auditmask=data["audit"];states=data["states"];N=len(states);block=data["block"]
 mean=states[trainmask].mean(0);std=np.maximum(states[trainmask].std(0),.01)
 state=torch.tensor((states-mean)/std,device="cuda");y=torch.tensor(data["target"],device="cuda");w=torch.tensor(data["weight"],device="cuda")
 dy=torch.tensor(data["dy"],device="cuda");dv=torch.tensor(data["dynvalid"],device="cuda",dtype=torch.float32)
 ids=np.flatnonzero(trainmask);win=cfg["window"]
 # All context stays in the same block for training and held-out checks.
 indices=[]
 for t in range(N):
  start=max(t-win+1,int(np.flatnonzero(block==block[t])[0]),1 if t>0 else 0)
  v=np.arange(start,t+1);v=np.pad(v,(win-len(v),0),mode="edge");indices.append(v)
 ix=torch.tensor(np.asarray(indices),device="cuda",dtype=torch.long)
 assert all(np.all(block[indices[t]]==block[t]) for t in range(N))
 assert np.all(trainmask[np.asarray(indices)[ids]])
 # Endpoint frame0 unknown cannot be supervised.
 assert data["weight"][0]==0 and not data["dynvalid"][0]
 if (data["weight"][ids]>0).sum()<4:raise RuntimeError("Insufficient measured weak anchors; no forced training")
 predictions=[];rawlogits=[];allstats=[]
 for seed in cfg["ensemble_seeds"]:
  np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  net=ContactFrameNet(states.shape[1]).cuda();optim=torch.optim.AdamW(net.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
  params=sum(p.numel() for p in net.parameters());rng=np.random.default_rng(seed);history=[]
  path=RUN/"artifacts"/("seed_"+str(seed));path.mkdir(exist_ok=False)
  torch.save({"model":net.state_dict(),"seed":seed},path/"initial.pt")
  stage("training",seed=seed,steps=cfg["steps"],parameters=params)
  for step in range(cfg["steps"]):
   net.train();chosen=rng.choice(ids,size=min(16,len(ids)),replace=False);j=torch.tensor(chosen,device="cuda");batch=ix[j]
   logit,pred=net(image[batch],point[batch],state[batch])
   bce=(nn.functional.binary_cross_entropy_with_logits(logit,y[j],reduction="none")*w[j]).mean()
   aux=(nn.functional.smooth_l1_loss(pred,dy[j],reduction="none").mean(-1)*dv[j]).mean()
   loss=bce+cfg["aux_weight"]*aux
   assert torch.isfinite(loss)
   optim.zero_grad();loss.backward();gn=nn.utils.clip_grad_norm_(net.parameters(),1.)
   assert torch.isfinite(gn);optim.step()
   history.append({"step":step+1,"loss":float(loss.detach()),"weighted_bce":float(bce.detach()),"aux":float(aux.detach())})
   if (step+1)%100==0:stage("training",seed=seed,step=step+1,loss=history[-1]["loss"])
  net.eval();ls=[];ds=[]
  with torch.inference_mode():
   for a in range(0,N,16):
    b=ix[a:a+16];l,d=net(image[b],point[b],state[b]);ls.append(l.cpu());ds.append(d.cpu())
  logits=torch.cat(ls);dyn=torch.cat(ds);pred=logits.sigmoid().numpy();rawlogits.append(logits.numpy());predictions.append(pred)
  torch.save({"model":net.state_dict(),"optimizer":optim.state_dict(),"seed":seed,"steps":cfg["steps"],"config":cfg,"state_mean":mean,"state_std":std,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(),"numpy_rng":rng.bit_generator.state,"state_dim":states.shape[1],"parameters":params},path/"last.pt")
  np.savez_compressed(path/"predictions.npz",logits=logits.numpy(),probability=pred,dynamics=dyn.numpy())
  save(path/"history.json",history)
  scores={}
  for label,mask in [("train",trainmask),("within_video_audit",auditmask)]:
   known=mask&(data["weight"]>0)
   values=nn.functional.binary_cross_entropy_with_logits(logits,torch.from_numpy(data["target"]),reduction="none").numpy()
   scores[label]={"frames":int(mask.sum()),"weighted_anchor_frames":int(known.sum()),"weighted_bce_per_all_frames":float((values[mask]*data["weight"][mask]).mean()),"bce_known_only":float(values[known].mean()) if known.any() else None,"mean_score":float(pred[mask].mean())}
  valid=data["dynvalid"];scores["train_dynamics_mae_m"]=float(np.mean(np.abs(dyn.numpy()[valid]-data["dy"][valid]))*.01) if valid.any() else None
  stats={"seed":seed,"steps":cfg["steps"],"parameters":params,"metrics":scores,"checkpoint_sha256":sha(path/"last.pt")};allstats.append(stats)
  del net,optim;torch.cuda.empty_cache()
 probs=np.stack(predictions);pmean=probs.mean(0);lo=probs.min(0);hi=probs.max(0)
 reliable=data["geometry_reliable"]&data["state_valid"]
 contact=(lo>=cfg["contact_display"])&reliable
 separate=(hi<=cfg["separation_display"])&reliable&(data["distance"]-2*data["sigma"]>.006)
 unknown=~(contact|separate);times=data["frame_time"]
 summary={"status":"trained_prototype_not_validated_contact_detector","episode":cfg["episode"],"frames":N,"runs":allstats,"candidate_contact":intervals(contact,times),"candidate_observed_separation":intervals(separate,times),"unknown":intervals(unknown,times),"counts":{"contact_candidate":int(contact.sum()),"observed_separation_candidate":int(separate.sum()),"unknown":int(unknown.sum())},"raw_score_threshold_intervals":{str(th):intervals(pmean>=th,times) for th in [.2,.5,.8]},"maximum_ensemble_spread":float((hi-lo).max()),"true_contact_accuracy":None,"failure_attempt_exclusion_rate":None,"inference_context":"causal 9-frame neural windows within fixed blocks; offline full-video tracker can use future pixels","no_task_success_labels":True,"no_held_or_lift_rule":True,"no_minimum_duration":True}
 save(RUN/"artifacts/model_summary.json",summary)
 np.savez_compressed(RUN/"artifacts/frame_predictions.npz",probabilities=probs,mean=pmean,low=lo,high=hi,contact_candidate=contact,observed_separation_candidate=separate,unknown=unknown,context_indices=np.asarray(indices))
 with (RUN/"artifacts/frame_predictions.csv").open("w",newline="",encoding="utf-8") as f:
  wr=csv.writer(f);wr.writerow(["frame","seconds","split","score_mean","score_min","score_max","state","weak_geometry_target","evidence_weight","distance_mm","sigma_mm","visible_tracks"])
  for t in range(N):
   wr.writerow([t,float(times[t]),"train" if trainmask[t] else "within_video_audit",float(pmean[t]),float(lo[t]),float(hi[t]),"contact_candidate" if contact[t] else "observed_separation_candidate" if separate[t] else "unknown",float(data["target"][t]),float(data["weight"][t]),float(data["distance"][t]*1000),float(data["sigma"][t]*1000),int(data["visible_counts"][t])])
 stage("training_complete",counts=summary["counts"])
if __name__=="__main__":
 try:main()
 except Exception as e:fail(e);raise SystemExit(1)

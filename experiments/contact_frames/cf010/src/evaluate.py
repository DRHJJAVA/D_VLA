from pathlib import Path
import numpy as np,torch,json,importlib.util,hashlib,traceback
from datetime import datetime
from zoneinfo import ZoneInfo
B=Path('/mnt/nvme1/drh/D_VLA');BASE=B/'experiments/contact_frames';R=BASE/'cf010';OLD=BASE/'cf008'
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def main():
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
 cfg=json.loads((R/'config.json').read_text());norm=json.loads((BASE/'cf003/normalization.json').read_text())
 mu=np.asarray(norm['mean'],dtype='float32');sd=np.asarray(norm['std'],dtype='float32')
 s=importlib.util.spec_from_file_location('cf010net',R/'src/model.py');mm=importlib.util.module_from_spec(s);s.loader.exec_module(mm)
 weights=json.loads((R/'evaluation_weights_locked.json').read_text())
 rows=[];causal=[]
 for ep in cfg['train_episodes']+cfg['fixed_review_episodes']:
  if ep in cfg['train_episodes']:src=OLD/'data'/ep/'artifacts/dataset.npz'
  elif ep in ['tomato_sauce_demo_3','alphabet_soup_demo_1']:src=BASE/'cf005_frozen_review_003'/ep/'artifacts/dataset.npz'
  else:src=BASE/'cf005/evaluation_all_train_frames'/ep/'artifacts/dataset.npz'
  d=dict(np.load(src));args=[torch.tensor(d[k],device='cuda')[None] for k in ['images','points']]
  args.append(torch.tensor((d['states']-mu)/sd,device='cuda')[None])
  ps=[];ns=[]
  for seed in cfg['seeds']:
   m=mm.ContactFrameNet(50).cuda();m.load_state_dict(torch.load(R/'runs/full_history'/('seed_'+str(seed))/'last.pt',map_location='cuda',weights_only=False)['model']);m.eval();assert not m.temporal.bidirectional
   with torch.no_grad():
    outputs=m(*args);e,a,n=outputs
    prefix=m(*[v[:,:60] for v in args])
    prefix_logits=[float((v[:,:60]-w).abs().max()) for v,w in zip(outputs,prefix)]
    prefix_prob=max(float((torch.sigmoid(v[:,:60])-torch.sigmoid(w)).abs().max()) for v,w in zip([e,n],[prefix[0],prefix[2]]))
    changed=[v.clone() for v in args]
    for v in changed:v[:,60:]=0
    fut=m(*changed)
    causal_diff=max(float((v[:,:60]-w[:,:60]).abs().max()) for v,w in zip(outputs,fut))
    assert causal_diff==0 and prefix_prob<=2e-5
    ps.append(torch.sigmoid(e[0]).cpu().numpy());ns.append(torch.sigmoid(n[0]).cpu().numpy())
    causal.append({'episode':ep,'seed':seed,'prefix_logit_diffs':prefix_logits,'prefix_probability_max_diff':prefix_prob,'future_perturbation_prefix_max_diff':causal_diff})
   del m
  p=np.stack(ps);near=np.stack(ns);reliable=d['geometry_reliable']&d['state_valid']
  separation=reliable&(d['distance']-2*d['sigma']>.006)
  predicted=(p.min(0)>=.8)&~separation;pos=predicted&reliable;neg=(near.max(0)<=.2)&separation
  out=R/'evaluation'/ep/'artifacts';out.mkdir(parents=True,exist_ok=True)
  if not (out/'dataset.npz').exists():(out/'dataset.npz').symlink_to(src.resolve())
  np.savez_compressed(out/'frame_predictions.npz',probabilities=p,proximity_probabilities=near,mean=p.mean(0),low=p.min(0),high=p.max(0),contact_candidate=pos,observed_separation_candidate=neg,unknown=~(pos|neg),predicted_contact=predicted,predicted_unknown=~(predicted|neg),geometry_reliable=reliable,geometry_separation_evidence=separation)
  rows.append({'episode':ep,'frames':len(pos),'strict_candidate':int(pos.sum()),'predicted_contact':int(predicted.sum()),'separation':int(neg.sum()),'strict_unknown':int((~(pos|neg)).sum())})
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in weights.items())
 save(R/'audits/temporal_causality.json',{'passed':True,'rows':causal,'note':'网络时间展开通过；原视频追踪预处理仍为离线'})
 save(R/'summary.json',{'rows':rows,'time':now(),'selection':'last600','human_ranges_training':False})
 save(R/'audits/evaluation_failure_resolved.json',{'time':now(),'original_failure_preserved':True,'retraining':False,'same_length_future_perturbation_passed':True})
 save(R/'stage_state.json',{'stage':'training_evaluation_complete','time':now()})
 print(json.dumps({'complete':True}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  (R/'evaluation_failure.log').write_text(traceback.format_exc())
  save(R/'evaluation_failure.json',{'type':type(e).__name__,'frames':[{'file':f.filename,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(e.__traceback__)]})
  raise SystemExit(1)

from adapter import *
def main():
 out=OUT/'audit';out.mkdir(exist_ok=True)
 model=load_model();front=Front();episodes=json.loads((RUN/'episodes.json').read_text());norm=json.loads((RUN/'normalization.json').read_text());std=np.asarray(norm['std']);mu=np.asarray(norm['mean']);rows=[];orig_diffs=[]
 for trial in [0,1,2]:
  w=OUT/'trials'/('trial_%03d'%trial);p=next((w/'cache/rgbd').glob('*.h5'));e=next(e for e in episodes if e['id']==p.stem)
  steps=[json.loads(x) for x in (w/'steps.jsonl').read_text().splitlines()]
  actions=[np.asarray(s['executed'],np.float32) for s in steps]
  words=torch.load(ROOT/'experiments/lcjp003/r2/v2/v001/cache'/(e['task']+'_words.pt'),map_location='cpu',weights_only=False)
  gen=torch.Generator().manual_seed(20261028+trial);plan=None
  with h5py.File(p,'r') as h:
   for t in range(101):
    d=runtime.move(front.inputs(h,t,words,actions[:t]),'cuda');noise=torch.randn((1,16,6),generator=gen).cuda()
    with torch.no_grad():
     enc=model.encode(d,step=3000);m,g,s=model.generate(enc,noise,plan);cls=g.argmax(-1)
     a=np.r_[m[0,0].cpu().numpy()*std+mu,float(cls[0,0])*2-1]
     orig_diffs.append(float(np.max(np.abs(a-np.asarray(steps[t]['action'])))))
     if t in [0,20,50,100]:
      for variant in ['uniform','shuffle','wrong']:
       ev=model.encode(d,score_override=variant,step=3000);mv,gv,sv=model.generate(ev,noise,plan);cv=gv.argmax(-1)
       av=np.r_[mv[0,0].cpu().numpy()*std+mu,float(cv[0,0])*2-1]
       rows.append({'trial':trial,'frame':t,'variant':variant,'motion_mae_change':float((mv-m).abs().mean()),'first_motion_raw_abs_max':float(np.abs(av[:6]-a[:6]).max()),'first_gripper_flip':bool(cv[0,0]!=cls[0,0]),'sequence_gripper_flip_rate':float((cv!=cls).float().mean()),'first_gripper_probability_change':float((gv.softmax(-1)[0,0]-g.softmax(-1)[0,0]).abs().max()),'lambda':float(enc['lambda'].mean()),'original_first_gripper':int(cls[0,0])})
     seq=torch.cat([m,cls.float().unsqueeze(-1)*2-1],-1);plan=torch.cat([seq[:,1:],seq[:,-1:]],1).detach()
 summary={'time':stamp(),'existing_trials_only':[0,1,2],'observation_frames_per_trial':101,'new_environment_steps':0,'original_action_reproduction_max_abs':max(orig_diffs),'paired_observation_and_noise_and_previous_plan':True,'rows':rows}
 (out/'score_replay.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:fail('audit_score_replay',e)

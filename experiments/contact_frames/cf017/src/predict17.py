import os,sys,json,pathlib,traceback,subprocess
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from common14 import *
import numpy as np,torch,h5py
from model14 import ContactMotionNet
C=ROOT/'experiments/contact_frames/cf014'
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.22)
def ranges(mask):
 idx=np.flatnonzero(mask);out=[]
 for x in idx:
  if out and x==out[-1][1]+1:out[-1][1]=int(x)
  else:out.append([int(x),int(x)])
 return out
def main():
 save(BASE/'prediction_source_locked.json',lock([p for p in (BASE/'src').glob('*.py') if p.is_file()]))
 verify(json.loads((BASE/'weights_locked.json').read_text())['files'])
 mn=json.loads((C/'motion_normalization.json').read_text());norm=json.loads((BASE.parent/'cf003/normalization.json').read_text())
 paths=[BASE/'config.json',BASE/'weights_locked.json',C/'motion_normalization.json',BASE.parent/'cf003/normalization.json']
 for c in CFG['cases']:paths += [dataset(c['id']),BASE/'features'/f"{c['id']}.npz",episode_config(c['id'])]
 save(BASE/'prediction_input_locked.json',lock(paths))
 cases=[]
 for c in CFG['cases']:
  ep=c['id'];stage('frozen_prediction_'+ep)
  with np.load(dataset(ep)) as d,np.load(BASE/'features'/f'{ep}.npz') as v:
   mo=(v['features']-np.array(mn['mean']))/np.array(mn['std']);mo[~v['valid']]=0
   ss=(d['states']-np.array(norm['mean']))/np.array(norm['std'])
   args=[torch.tensor(z,dtype=torch.float32,device='cuda')[None] for z in [d['images'],d['points'],ss,mo]]
  probs=[];near=[]
  for seed in CFG['seeds']:
   net=ContactMotionNet(True).cuda()
   net.load_state_dict(torch.load(C/'runs/with_motion'/f'seed_{seed}'/'last.pt',map_location='cuda',weights_only=False)['model']);net.eval()
   with torch.no_grad():out=net(*args)
   probs.append(out[0].sigmoid()[0].cpu().numpy());near.append(out[2].sigmoid()[0].cpu().numpy())
  p=np.asarray(probs);n=np.asarray(near);dest=BASE/'evaluation'/ep;dest.mkdir(exist_ok=True)
  np.savez_compressed(dest/'predictions.npz',probabilities=p,proximity=n)
  cases.append(dict(id=ep,name=c['display'],fps=20,n=p.shape[1],src=f'图示/cf017_contact/{ep}_clean.mp4',human=None,negative=None,mean=p.mean(0).tolist(),low=p.min(0).tolist(),high=p.max(0).tolist(),seeds=p.tolist(),near_mean=n.mean(0).tolist(),ranges=ranges(p.mean(0)>=.8),note='未训练的新任务；首帧表面由整段实际夹爪轨迹空间并集选择，存在输入分布变化。无人工接触标签，候选不是准确率。'))
 save(BASE/'predictions_frozen.json',dict(time=now(),files=lock(list((BASE/'evaluation').glob('*/predictions.npz'))),weights=lock(list((C/'runs/with_motion').glob('*/last.pt'))),manual_ranges_read=False))
 # Existing demo1 review only after all new predictions freeze.
 ep='alphabet_soup_demo_1'
 with np.load(C/'evaluation'/ep/'predictions.npz') as z:p=z['with_motion']
 oldpayload=json.loads((ROOT/'docs/图示/cf014_contact/data.json').read_text());old=next(x for x in oldpayload['cases'] if x['id']==ep)
 cases.append(dict(id=ep,name='罐头demo1 · 原运输低分复核',fps=old['fps'],n=old['n'],src=old['src'],human=[85,166],negative=None,mean=p.mean(0).tolist(),low=p.min(0).tolist(),high=p.max(0).tolist(),seeds=p.tolist(),ranges=ranges(p.mean(0)>=.8),note='CF014原保存预测，未重训。117–137帧均分0.282；几何仍贴近、跟踪可见，低分不证明脱离。'))
 assets=ROOT/'docs/图示/cf017_contact';assets.mkdir(parents=True,exist_ok=True)
 save(assets/'data.json',dict(time=now(),cases=cases,model='CF014 with_motion last600 seeds2026/2027/2028',new_task_count=3,no_training=True))
 save(BASE/'evaluation/summary.json',dict(time=now(),cases=[dict(id=c['id'],frames=c['n'],candidate_ranges=c['ranges'],high_frames=int(np.sum(np.asarray(c['mean'])>=.8))) for c in cases],new_task_manual_labels=False))
 stage('rendering_clean_review_videos')
 for c in CFG['cases']:
  ep=c['id'];out=assets/f'{ep}_clean.mp4';conf=json.loads(episode_config(ep).read_text())
  if out.exists():raise RuntimeError('video_exists')
  with h5py.File(conf['rgbd'],'r') as h,open(BASE/'cases'/ep/'encode.log','w') as log:
   a=h['agentview/rgb'];b=h['robot0_eye_in_hand/rgb'];H,W=a.shape[1:3]
   cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W*2}x{H}','-r','20','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
   proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
   for i in range(len(a)):proc.stdin.write(np.concatenate([a[i],b[i]],axis=1).tobytes())
   proc.stdin.close();rc=proc.wait()
   if rc:raise RuntimeError('ffmpeg_failed_'+ep)
 (ROOT/'docs/CF017_跨任务接触预测查看.html').write_bytes((BASE/'src/review17.html').read_bytes())
 verify(json.loads((BASE/'prediction_input_locked.json').read_text()))
 verify(json.loads((BASE/'weights_locked.json').read_text())['files'])
 stage('prediction_page_ready')
if __name__=='__main__':
 try:main()
 except Exception as e:
  save(BASE/'failure_predict.json',dict(time=now(),type=type(e).__name__,message=str(e),stack=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_tb(e.__traceback__)]));stage('prediction_failed');raise SystemExit(1)

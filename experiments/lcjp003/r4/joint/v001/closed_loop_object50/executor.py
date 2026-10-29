from adapter import *
import fcntl,subprocess,cv2,html,imageio_ffmpeg
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
ASSETS=ROOT/'docs/LCJP003_R4_object50_assets'
PAGE=ROOT/'docs/LCJP003_R4_Object50闭环测试.html'
NAMES={'alphabet_soup':'字母汤罐','bbq_sauce':'烧烤酱','butter':'黄油','chocolate_pudding':'巧克力布丁','cream_cheese':'奶油奶酪','ketchup':'番茄酱','milk':'牛奶','orange_juice':'橙汁','salad_dressing':'沙拉酱','tomato_sauce':'意面番茄酱'}
def title(task):
 for key,value in NAMES.items():
  if 'pick_up_the_'+key+'_and' in task:return '拿起'+value+'并放入篮子'
 return task
def page(rows,current=None):
 success=sum(int(x['success']) for x in rows)
 chosen=[x for x in rows if x['success']][:3]+[x for x in rows if not x['success']][:3]
 if not chosen:chosen=rows[:6]
 cards=''.join('<article><h2>'+html.escape(title(x['task']))+' · '+('成功' if x['success'] else '失败')+'</h2><p>试验 '+str(x['trial']+1)+' · 初始状态 '+str(x['initial_state_index'])+' · '+str(x['steps'])+' 步</p><video controls preload="metadata" src="LCJP003_R4_object50_assets/'+x['video']+'"></video></article>' for x in chosen)
 table=''.join('<tr><td>'+str(x['trial']+1)+'</td><td>'+html.escape(title(x['task']))+'</td><td>'+str(x['initial_state_index'])+'</td><td>'+('成功' if x['success'] else '失败')+'</td><td>'+str(x['steps'])+'</td><td><a href="LCJP003_R4_object50_assets/'+x['video']+'">视频</a></td></tr>' for x in rows)
 body='<!doctype html><html lang="zh"><meta charset="utf-8"><title>R4 Object 50次闭环</title><style>body{background:#101820;color:#edf2f7;font:16px system-ui;margin:32px auto;max-width:1200px}h1{font-size:30px}p{line-height:1.7;color:#c8d7e1}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{background:#1b2834;padding:16px;border-radius:12px}video{width:100%}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #34424d}a{color:#7dd3fc}</style><h1>R4 · LIBERO-Object 闭环测试</h1><p>最终3000次更新权重。10个任务各5个官方初始状态，共50次；每步重新观测，仅执行预测首动作。无在线SAM、无专家动作输入。</p><h2>已完成 '+str(len(rows))+'/50 · 成功 '+str(success)+' · 当前成功率 '+(f'{success/len(rows):.1%}' if rows else '尚无完成试验')+'</h2><p>成功由仿真环境 check_success 判断；未完成任务达到600步记为失败。当前结果不是完整500次基准，也不是新物体形状泛化测试。</p><p>左：外部相机；右：腕部相机。白色轮廓为模型预测目标；目标内保留最高80%可见点，在保留点内重新排名：最高5%红、下一5%橙、80–90百分位黄、60–80绿、其余蓝。颜色是相对排名；完整低分、分割和评分保存在远程。视频20帧/秒播放，实际推理耗时另行统计。</p><div class="grid">'+cards+'</div><h2>全部已完成试验</h2><table><tr><th>编号</th><th>任务</th><th>初始状态</th><th>结果</th><th>控制步</th><th>查看</th></tr>'+table+'</table><p>更新时间 '+stamp()+'</p></html>'
 tmp=PAGE.with_suffix('.tmp');tmp.write_text(body);tmp.replace(PAGE)
def draw(rgb,t,enc,trial,step):
 images=[rgb[c+'/rgb'][t].copy() for c in ['agentview','robot0_eye_in_hand']]
 scores=enc['point_score'].detach().cpu().numpy().reshape(-1)
 pts=enc['points'];uv=pts['uv'].detach().cpu().numpy();cam=pts['cam'].detach().cpu().numpy().reshape(-1)
 valid=pts['valid'].detach().cpu().numpy().reshape(-1).astype(bool)
 logits=enc['seg']['logits'].detach().cpu().numpy()
 masks=(logits[:,0]>0) if logits.ndim==4 else (logits>0)
 selected=np.flatnonzero(valid);selected=selected[np.argsort(scores[selected],kind='stable')]
 selected=selected[-max(1,int(np.ceil(.8*len(selected)))):] if len(selected) else selected
 for ci,im in enumerate(images):
  ma=masks[ci].astype(np.uint8);ma=cv2.resize(ma,(256,256),interpolation=cv2.INTER_NEAREST)
  contours,_=cv2.findContours(ma,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,contours,-1,(255,255,255),1)
 for rank,k in enumerate(selected):
  ci=int(cam[k]);x,y=np.rint(uv[k]).astype(int)
  if ci not in [0,1] or not (0<=x<256 and 0<=y<256):continue
  q=(rank+.5)/len(selected);color=(239,68,68) if q>=.95 else (249,145,40) if q>=.9 else (250,214,55) if q>=.8 else (65,195,110) if q>=.6 else (55,135,238)
  cv2.circle(images[ci],(x,y),1,color,-1)
 canvas=np.zeros((300,512,3),np.uint8);canvas[:256]=np.concatenate(images,axis=1)
 cv2.putText(canvas,f'Trial {trial+1:02d} | frame {step:03d} | external / wrist',(8,276),cv2.FONT_HERSHEY_SIMPLEX,.43,(235,235,235),1,cv2.LINE_AA)
 cv2.putText(canvas,'top80% retained | relative ranks | model mask: white',(8,294),cv2.FONT_HERSHEY_SIMPLEX,.38,(200,210,220),1,cv2.LINE_AA)
 return canvas
def main():
 OUT.mkdir(exist_ok=True);ASSETS.mkdir(exist_ok=True)
 lock=open(OUT/'executor.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((OUT/'ready.json').read_text())['passed']
 protocol=json.loads((OUT/'protocol.json').read_text())
 assert sha(RUN/'checkpoints/step3000.pt')==protocol['checkpoint_sha256']
 model=load_model();front=Front();renderer=render_module()
 rows=json.loads((OUT/'results.json').read_text())['trials'] if (OUT/'results.json').exists() else []
 episodes=json.loads((RUN/'episodes.json').read_text());tasks=sorted(set(e['task'] for e in episodes))
 vendor=pathlib.Path('/home/drh/project/ECV-VLA/vendor/LIBERO/libero/libero')
 alltimes=[]
 for trial in range(len(rows),50):
  if (OUT/'STOP').exists():save('state.json',{'status':'paused','time':stamp(),'completed_trials':len(rows)});return
  task=tasks[trial//5];initial=trial%5;e=next(e for e in episodes if e['task']==task and e['episode']=='demo_0')
  work=OUT/'trials'/f'trial_{trial:03d}'
  if work.exists():raise RuntimeError('partial_trial_preserved_requires_review')
  work.mkdir(parents=True)
  render=renderer.LiveRender(e['id'],work,capacity=602)
  initpath=vendor/'init_files/libero_object'/(task.removesuffix('_demo')+'.pruned_init')
  states=torch.load(initpath,map_location='cpu',weights_only=False)
  render.env.set_init_state(states[initial])
  for _ in range(5):render.env.step(np.zeros(7))
  render.capture(0)
  words=torch.load(ROOT/'experiments/lcjp003/r2/v2/v001/cache'/(task+'_words.pt'),map_location='cpu',weights_only=False)
  acts=[];plan=None;generator=torch.Generator().manual_seed(20261028+trial)
  video=f'trial_{trial:03d}.mp4'
  ff=subprocess.Popen([FFMPEG,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','512x300','-r','20','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(ASSETS/video)],stdin=subprocess.PIPE,stderr=open(work/'encoder.log','wb'))
  success=False;done=False;started=time.perf_counter()
  for step in range(600):
   if (OUT/'STOP').exists():ff.stdin.close();ff.wait();render.close();save('state.json',{'status':'paused_partial_trial','trial':trial,'step':step,'time':stamp()});return
   tic=time.perf_counter();d=front.inputs(render.out,step,words,acts);frontsec=time.perf_counter()-tic
   tic2=time.perf_counter();a,plan,enc=predict(model,d,generator,plan);torch.cuda.synchronize();policysec=time.perf_counter()-tic2
   frame=draw(render.out,step,enc,trial,step);ff.stdin.write(frame.tobytes())
   np.savez_compressed(work/f'points_{step:04d}.npz',xyz=enc['points']['xyz'].detach().cpu().numpy(),uv=enc['points']['uv'].detach().cpu().numpy(),camera=enc['points']['cam'].detach().cpu().numpy(),valid=enc['points']['valid'].detach().cpu().numpy(),score=enc['point_score'].detach().cpu().numpy(),mask_logits=enc['seg']['logits'].detach().cpu().numpy())
   executed=np.clip(a,-1,1);_,done=render.step(executed,step+1);acts.append(executed)
   success=bool(render.env.check_success());seconds=time.perf_counter()-tic;alltimes.append(seconds)
   with (work/'steps.jsonl').open('a') as f:f.write(json.dumps({'step':step,'front_seconds':frontsec,'policy_seconds':policysec,'total_seconds':seconds,'action':a.tolist(),'executed':executed.tolist(),'success':success})+'\n')
   if step%10==0:save('state.json',{'status':'evaluating','trial':trial,'task':task,'initial_state_index':initial,'step':step+1,'completed_trials':len(rows),'successes':sum(int(x['success']) for x in rows),'recent_step_seconds':float(np.mean(alltimes[-20:])),'time':stamp()})
   if len(alltimes)==20:save('eta20.json',{'mean_step_seconds':float(np.mean(alltimes)),'worst_case_remaining_seconds':float(np.mean(alltimes))*(30000-20),'time':stamp(),'includes_render_and_video':True})
   if success or done:break
  ff.stdin.close();assert ff.wait()==0
  render.close()
  check=subprocess.run([FFMPEG,'-v','error','-i',str(ASSETS/video),'-progress','pipe:1','-f','null','-'],capture_output=True,text=True);assert check.returncode==0
  decoded=[int(x.split('=')[1]) for x in check.stdout.splitlines() if x.startswith('frame=')]
  assert decoded and decoded[-1]==len(acts)
  result={'trial':trial,'task':task,'initial_state_index':initial,'success':success,'steps':len(acts),'termination':'success' if success else 'environment_done' if done else 'max_steps','elapsed_seconds':time.perf_counter()-started,'video':video,'decoded_frames':len(acts),'time':stamp()}
  (work/'result.json').write_text(json.dumps(result,indent=2));rows.append(result)
  summary={'completed':len(rows),'planned':50,'successes':sum(int(x['success']) for x in rows),'success_rate':sum(int(x['success']) for x in rows)/len(rows),'trials':rows,'time':stamp()}
  save('results.json',summary);page(rows)
  if trial==0:save('first_video.json',result)
 assert sha(RUN/'checkpoints/step3000.pt')==protocol['checkpoint_sha256']
 save('complete.json',summary);save('state.json',{'status':'complete','completed_trials':50,'successes':summary['successes'],'time':stamp()})
 with (ROOT/'docs/LCJP003_状态条件三维部位评分与DriftingVLA设计方案.md').open('a') as f:f.write('\n\n### '+stamp()+' R4 Object50闭环完成\n成功 '+str(summary['successes'])+'/50。页面：[Object50闭环测试](LCJP003_R4_Object50闭环测试.html)。全部试验视频/评分/原始RGBD与失败保留在closed_loop_object50。此为10任务各5初始状态的小规模测试，不自动扩展到500次。\n')
if __name__=='__main__':
 try:main()
 except Exception as e:fail('closed_loop',e);raise SystemExit(1)

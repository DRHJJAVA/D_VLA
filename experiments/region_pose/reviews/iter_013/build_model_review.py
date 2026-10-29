from pathlib import Path
import os,sys,json,hashlib,importlib.util
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,h5py,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/region_pose/iter_013';V=B.parent/'reviews/iter_013'
sys.path[:0]=[str(B/'src'),str(R/'src/v3'),str(R/'src')]
from model import RegionPoseNet
from stage2_extract_dino import model as dino_model,preproc
from prepare_samples import buildgraph
from build_point_graphs import fps
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False))
def main():
 assert json.loads((B/'stage_state.json').read_text())['stage']=='complete'
 assert not (V/'locked.json').exists()
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
 for d in ['inputs','graphs','predictions']: (V/d).mkdir(exist_ok=True)
 images=R/'docs/图示/第013轮真实预测';images.mkdir(exist_ok=False)
 refs={r['id']:r for r in json.loads((B.parent/'iter_003/evidence/base/protocol.json').read_text())['episodes']}
 rows=json.loads((B/'manifest.json').read_text())['rows'];lookup={r['key']:r for r in rows}
 cases=[];netd=None
 for split in ['val','train']:
  for c in json.loads((B.parent/'iter_012'/split/'summary.json').read_text())['cases']:
   key=c['event']+'_query'+str(c['query']);sample=B/'samples'/(key+'.npz');gr=B/'graphs'/(key+'.npz')
   accepted=sample.exists()
   if not accepted:
    # No target arrays read; this extra input is prepared solely from causal query observations.
    z=np.load(B.parent/'iter_012'/split/(key+'.npz'));world=z['query_xyz'];full=z['query_pixel'].astype(np.int64)
    sel=fps(world,2048) if len(world)>2048 else np.arange(len(world));points=world[sel];pixel=full[sel]
    m=np.load(B.parent/'iter_003/evidence/motion'/(c['id']+'.npz'));H=m['hand_to_world'][c['query']];q=m['measured_finger_q'][c['query']]
    with h5py.File(refs[c['id']]['rgbd'],'r') as f:
     rgb=f['agentview/rgb'][c['query']];Tc=f['agentview/camera_to_world'][c['query']]
     center=points.mean(0);xyz=(points-center)@Tc[:3,:3];Rh=Tc[:3,:3].T@H[:3,:3];th=(H[:3,3]-center)@Tc[:3,:3];pose=np.r_[th/.1,Rh[:,0],Rh[:,1],(q[0]-q[1]-.001)/.1]
     im=np.full_like(rgb,127);im[full[:,1],full[:,0]]=rgb[full[:,1],full[:,0]];x1,y1=full.min(0);x2,y2=full.max(0)+1
     if netd is None:netd=dino_model()
     with torch.inference_mode():patch=netd.forward_features(preproc(im[None,y1:y2,x1:x2]))[:,1:,:].reshape(1,32,32,384).permute(0,3,1,2)
     gx=(pixel[:,0]-x1+.5)/(x2-x1)*2-1;gy=(pixel[:,1]-y1+.5)/(y2-y1)*2-1;grid=torch.as_tensor(np.stack([gx,gy],1)[None,None],dtype=patch.dtype,device=patch.device)
     with torch.inference_mode():feat=F.grid_sample(patch,grid,align_corners=False).squeeze(0).squeeze(1).T.cpu().numpy()
     sample=V/'inputs'/(key+'.npz');gr=V/'graphs'/(key+'.npz')
     np.savez_compressed(sample,xyz=xyz.astype(np.float32),dino_feat=feat.astype(np.float16),pose=pose.astype(np.float32),hand_R=Rh.astype(np.float32),hand_t=th.astype(np.float32),pixel=pixel.astype(np.int32),input_rgb_pixel=full.astype(np.int32),input_crop_box=np.array([x1,y1,x2,y2],np.int32))
     np.savez_compressed(gr,**buildgraph(xyz))
   cases.append({'key':key,'episode':c['id'],'split':split,'query':c['query'],'sample':str(sample),'graph':str(gr),'sample_sha256':sha(sample),'graph_sha256':sha(gr),'evaluated_with_sufficient_automatic_targets':accepted})
 if netd is not None:del netd
 torch.cuda.empty_cache()
 checkpoint={}
 for variant in ['object_only','pose_conditioned']:
  path=B/'runs'/variant/'best.pt';checkpoint[variant]=sha(path);ck=torch.load(path,map_location='cuda',weights_only=False);net=RegionPoseNet().cuda().eval();net.load_state_dict(ck['state'])
  with torch.inference_mode():
   for c in cases:
    z=np.load(c['sample']);g=np.load(c['graph'])
    inp=[torch.as_tensor(z[k],dtype=torch.float32,device='cuda') for k in ['xyz','dino_feat']]
    graph={k:torch.as_tensor(g[k],dtype=torch.float32 if k.endswith('weight') else torch.long,device='cuda') for k in g.files}
    a,b=net(*inp,graph,*[torch.as_tensor(z[k],dtype=torch.float32,device='cuda') for k in ['pose','hand_R','hand_t']]);score=(a if variant=='object_only' else b).sigmoid().cpu().numpy()
    np.savez_compressed(V/'predictions'/(c['key']+'_'+variant+'.npz'),score=score,pixel=z['pixel'],checkpoint_sha256=np.asarray(checkpoint[variant]),input_sha256=np.asarray(c['sample_sha256']))
  del net,ck;torch.cuda.empty_cache()
 parts=['<!doctype html><meta charset="utf-8"><title>第013轮 单物体真实模型预测</title><style>body{font:16px sans-serif;max-width:1350px;margin:30px auto;background:#f3f5f8;color:#182130}section{background:white;padding:24px;margin:25px 0}.row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}img{width:100%}figure{margin:8px}figcaption{padding:9px}</style><h1>第013轮：单物体真实模型预测</h1><p>左：实际完整RGB物体输入；中：基础模型；右：加入当前夹爪位姿的条件模型。黄色为模型对全部输入点评分的前20%，不是自动目标、不是接触真值，也不是校准成功概率。点云和当前夹爪位姿同样输入模型。</p><p>两组各30轮，从头训练。本轮有效6训练事件、2旧验证事件；白黄杯验证自动目标只有1个正点，不纳入定量评估，但其真实模型输出仍完整展示。番茄酱验证缺少持物证据，尚未覆盖。第二步未验收。</p>']
 for c in cases:
  z=np.load(c['sample']);pixel=z['pixel'].astype(np.int64);full=z['input_rgb_pixel'].astype(np.int64);box=z['input_crop_box']
  with h5py.File(refs[c['episode']]['rgbd'],'r') as f:rgb=f['agentview/rgb'][c['query']]
  im=np.full_like(rgb,127);im[full[:,1],full[:,0]]=rgb[full[:,1],full[:,0]]
  parts.append('<section><h2>'+c['episode']+' · '+('旧验证' if c['split']=='val' else '训练示例')+' · 查询帧 '+str(c['query'])+'</h2>')
  if not c['evaluated_with_sufficient_automatic_targets']:parts.append('<p><b>自动目标不足，未计入本轮指标；下图仍是模型真实推理，不能因此称效果已正确。</b></p>')
  parts.append('<div class="row">')
  for variant,label in [('input','完整单物体RGB输入'),('object_only','基础模型：前20%评分'),('pose_conditioned','当前位姿条件模型：前20%评分')]:
   pic=Image.fromarray(im)
   if variant!='input':
    score=np.load(V/'predictions'/(c['key']+'_'+variant+'.npz'))['score'];chosen=np.argsort(-score,kind='stable')[:int(np.ceil(.2*len(score)))];dr=ImageDraw.Draw(pic)
    for u,v in pixel[chosen]:dr.ellipse((int(u)-1,int(v)-1,int(u)+1,int(v)+1),fill='#ffd52a')
   name=c['key']+'_'+variant+'.png';pic.crop(tuple(map(int,box))).resize((448,448)).save(images/name)
   parts.append('<figure><img src="图示/第013轮真实预测/'+name+'"><figcaption>'+label+'</figcaption></figure>')
  parts.append('</div></section>')
 page=R/'docs/第013轮_真实模型预测与完整输入.html';page.write_text('\n'.join(parts))
 dump(V/'inference_audit.json',{'cases':cases,'checkpoints':checkpoint,'prediction_arrays':len(cases)*2,'future_not_used_in_inference':True,'user_reviewed':False})
 files=[p for p in V.rglob('*') if p.is_file() and p.name!='locked.json']+[page]+list(images.glob('*.png'))
 dump(V/'locked.json',{str(p.relative_to(R)):sha(p) for p in files})
 print('INFERENCE_REVIEW_COMPLETE',len(cases),len(cases)*3,flush=True)
if __name__=='__main__':main()

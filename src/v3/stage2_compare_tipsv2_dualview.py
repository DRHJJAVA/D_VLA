"""Frozen encoder comparison using RGB-D same-time dual-view consistency only.
Not a grasp model, object ground truth, or contact evaluation. No held-out/reserved reads.
"""
from pathlib import Path
import os
os.environ['XFORMERS_DISABLED']='1'
import argparse,hashlib,json,sys,time,importlib.util
import cv2,h5py,numpy as np,torch,torch.nn.functional as F
R=Path('/mnt/nvme1/drh/D_VLA')
RUN=R/'audits/v3/tipsv2_dualview_pilot_v1'
SHA='bf10a73765b048edfbfcdf1f7e6ca8292542baae'
TRAIN={'tomato_sauce_demo_3':[0,45,75,110,150],
       'porcelain_mug_demo_4':[0,36,57,70,90],
       'white_yellow_mug_demo_2':[0,45,61,75,100]}
def save_json(path,obj):
 text=json.dumps(obj,indent=2,ensure_ascii=False)
 if path.exists():
  assert path.read_text()==text, str(path)
 else:path.write_text(text)
def bins(rgb):
 h=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
 return np.ravel_multi_index(np.stack((np.minimum(h[:,:,0]//15,11),np.minimum(h[:,:,1]//64,3),np.minimum(h[:,:,2]//64,3)),axis=-1).reshape(-1,3).T,(12,4,4)).reshape(h.shape[:2])
def lut(rgb,mask,box):
 b=bins(rgb);x1,y1,x2,y2=box;ring=np.zeros(mask.shape,bool)
 ring[max(0,y1-20):min(len(mask),y2+20),max(0,x1-20):min(mask.shape[1],x2+20)]=True;ring[mask]=False
 a=np.bincount(b[mask],minlength=192).astype(float);c=np.bincount(b[ring],minlength=192).astype(float)
 return np.log((a+.5)/(a.sum()+96)/(c+.5)*(c.sum()+96))
def mask_now(rgb,depth,box,ll):
 x,y,w,h=map(int,box);x1=max(0,x);y1=max(0,y);x2=min(rgb.shape[1],x+w);y2=min(rgb.shape[0],y+h)
 m=np.zeros(depth.shape,bool)
 if x2>x1 and y2>y1:m[y1:y2,x1:x2]=(ll[bins(rgb[y1:y2,x1:x2])]>0)
 return m&np.isfinite(depth)&(depth>.05)
def xyz(uv,dep,K,T):
 p=np.rint(uv).astype(int);z=dep[p[:,1],p[:,0]]
 a=np.stack(((uv[:,0]-K[0,2])*z/K[0,0],(uv[:,1]-K[1,2])*z/K[1,1],z),1)
 return a@T[:3,:3].T+T[:3,3]
def project(points,K,T):
 c=(points-T[:3,3])@T[:3,:3];z=c[:,2]
 u=np.stack((K[0,0]*c[:,0]/np.maximum(z,.001)+K[0,2],K[1,1]*c[:,1]/np.maximum(z,.001)+K[1,2]),1)
 return u,z
def box_crop(uv,shape):
 lo=uv.min(0);hi=uv.max(0)+1;center=(hi+lo)/2;size=np.maximum((hi-lo)*1.25,64)
 x1,y1=np.maximum(np.floor(center-size/2),0).astype(int)
 x2,y2=np.minimum(np.ceil(center+size/2),[shape[1],shape[0]]).astype(int)
 return [int(x1),int(y1),int(x2),int(y2)]
def grid(box):
 x1,y1,x2,y2=box
 u,v=np.meshgrid((np.arange(32)+.5)*(x2-x1)/32+x1-.5,(np.arange(32)+.5)*(y2-y1)/32+y1-.5)
 return np.stack((u,v),-1).reshape(-1,2)
def manifest():
 rows=[]
 for split in ['train','val']:
  for cp in sorted((R/'data/v3/proposals/foreground3d_components'/split).glob('*.npz')):
   if split=='train' and cp.stem not in TRAIN:continue
   with np.load(cp) as z:rp=Path(str(z['rgbd_path']))
   with h5py.File(rp,'r') as f:n=len(f['agentview/rgb'])
   frames=TRAIN[cp.stem] if split=='train' else sorted(set([0]+[round((n-1)*q) for q in [.25,.4,.55,.75]]))
   assert all(0<=t<n for t in frames)
   rows.append(dict(id=cp.stem,split=split,component=str(cp),rgbd=str(rp),frames=frames))
 assert len(rows)==7 and sum(x['split']=='val' for x in rows)==4
 return dict(version=1,episodes=rows,feature_size=448,grid=32,crop_padding=1.25,minimum_crop=64,
   source_mask='first frame automatic 3D component; later HSV support inside MIL candidate',
   visible_threshold_m=.015,eval_thresholds_m=[.008,.015,.030],minimum_queries=16,
   target_candidates='all valid-depth patch centers in projected object crop; no pairwise geometry in feature matching',
   evaluation='same timestamp two cameras, calibrated RGB-D self-consistency; NOT identity/contact ground truth',
   model_input='local RGB crops only; RGB-D/camera calibration only construct crops and independent consistency diagnostics',
   no_contacts=True,no_actions=True,no_names_or_text_inputs=True,no_manual_labels=True,no_reserved=True)
def pair(row,t):
 with np.load(row['component']) as z:pix=z['pixel'];bb=z['component_box'].astype(int)
 tp=R/'data/v3/tracking/mil_448_v1'/row['split']/(row['id']+'.npz')
 with np.load(tp) as z:boxes=z['boxes']
 with h5py.File(row['rgbd'],'r') as f:
  a=f['agentview'];w=f['robot0_eye_in_hand']
  first=a['rgb'][0];init=np.zeros(first.shape[:2],bool);init[pix[:,1],pix[:,0]]=True
  rgb=a['rgb'][t];dep=a['depth_m'][t];K=a['K'][t];T=a['camera_to_world'][t]
  wrgb=w['rgb'][t];wd=w['depth_m'][t];wK=w['K'][t];wT=w['camera_to_world'][t]
  m=init if t==0 else mask_now(rgb,dep,boxes[t],lut(first,init,bb))
 if m.sum()<16:return None,'too_few_source_candidate_pixels'
 sb=box_crop(np.argwhere(m)[:,::-1],dep.shape)
 puv=np.argwhere(m)[:,::-1].astype(float)
 pp=xyz(puv,dep,K,T);uv,z=project(pp,wK,wT)
 ii=(z>.05)&(uv[:,0]>=0)&(uv[:,0]<wd.shape[1]-1)&(uv[:,1]>=0)&(uv[:,1]<wd.shape[0]-1)
 if ii.sum()<16:return None,'object_projection_outside_wrist'
 tb=box_crop(uv[ii],wd.shape)
 su=grid(sb);si=np.rint(su).astype(int);source_valid=m[si[:,1],si[:,0]]
 sd=dep[si[:,1],si[:,0]]
 # Avoid depth-discontinuity centers independent of encoder.
 local_range=cv2.dilate(dep,np.ones((3,3),np.uint8))-cv2.erode(dep,np.ones((3,3),np.uint8))
 source_valid&=(local_range[si[:,1],si[:,0]]<.01)&np.isfinite(sd)&(sd>.05)
 sx=xyz(su,dep,K,T);tu,tz=project(sx,wK,wT)
 inside=(tz>.05)&(tu[:,0]>=0)&(tu[:,0]<wd.shape[1]-1)&(tu[:,1]>=0)&(tu[:,1]<wd.shape[0]-1)
 delta=np.full(1024,np.inf);idx=np.flatnonzero(inside);tpix=np.rint(tu[idx]).astype(int)
 delta[idx]=np.abs(wd[tpix[:,1],tpix[:,0]]-tz[idx])
 src=np.flatnonzero(source_valid&inside&(delta<.015))
 gu=grid(tb);gi=np.rint(gu).astype(int);gd=wd[gi[:,1],gi[:,0]]
 dst=np.flatnonzero(np.isfinite(gd)&(gd>.05))
 if len(src)<16 or len(dst)<16:return None,'insufficient_common_visible_surface'
 tx=xyz(gu,wd,wK,wT)
 x1,y1,x2,y2=sb;u1,v1,u2,v2=tb
 return dict(images=[rgb[y1:y2,x1:x2],wrgb[v1:v2,u1:u2]],source_box=sb,target_box=tb,
   source_indices=src,target_indices=dst,source_xyz=sx[src],target_xyz=tx[dst],
   source_uv=su[src],target_uv=gu[dst],expected_target_uv=tu[src],source_grid=su,target_grid=gu),None
def model(kind):
 if kind=='dinov2_s':
  from stage2_extract_dino import model as getmodel
  return getmodel()
 if kind=='dinov2_b':
  import timm
  net=timm.create_model('vit_base_patch14_dinov2',pretrained=False,dynamic_img_size=True)
  w=torch.load(R/'weights/dinov2_b14/dinov2_vitb14_pretrain.pth',map_location='cpu',weights_only=True)
  state=net.load_state_dict(w,strict=False);assert not state.missing_keys and state.unexpected_keys==['mask_token']
  return net.eval().cuda()
 spec=importlib.util.spec_from_file_location('tips_image_encoder',R/'third_party/tips_encoder_ablation'/SHA/'pytorch/image_encoder.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 net=mod.vit_base(img_size=448,patch_size=14,ffn_layer='mlp',block_chunks=0,init_values=1.,interpolate_antialias=True,interpolate_offset=0.)
 with np.load(R/'weights/tipsv2_b14/tips_v2_oss_b14_vision.npz',allow_pickle=False) as w:
  net.load_state_dict({k:torch.from_numpy(w[k]) for k in w.files},strict=True)
 return net.eval().cuda()
def features(net,kind,images):
 out=[]
 for im in images:
  x=torch.from_numpy(np.ascontiguousarray(im)).permute(2,0,1)[None].cuda().float()/255
  x=F.interpolate(x,size=(448,448),mode='bilinear',align_corners=False,antialias=True)
  if kind.startswith('dinov2'):
   x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
  with torch.inference_mode():
   q=net.forward_features(x)
   f=q['x_norm_patchtokens'][0] if kind=='tipsv2_b' else q[0,1:]
   assert len(f)==1024 and torch.isfinite(f).all()
   out.append(F.normalize(f.float(),dim=-1).cpu().numpy())
 return out
def evaluate(net,kind,row,t):
 d,reason=pair(row,t)
 result=dict(id=row['id'],split=row['split'],frame=t,model=kind,abstain=reason)
 if d is None:return result
 fa,fb=features(net,kind,d['images'])
 sim=fa[d['source_indices']]@fb[d['target_indices']].T
 selected=sim.argmax(1);reverse=sim.argmax(0);mutual=reverse[selected]==np.arange(len(selected))
 errors=np.linalg.norm(d['source_xyz']-d['target_xyz'][selected],axis=1)
 dist=np.linalg.norm(d['source_xyz'][:,None]-d['target_xyz'][None],axis=2)
 result.update(queries=len(selected),target_candidates=len(d['target_indices']),
    source_box=d['source_box'],target_box=d['target_box'],median_error_mm=float(np.median(errors)*1000),
    mutual_fraction=float(mutual.mean()),mutual_queries=int(mutual.sum()))
 for mm in [8,15,30]:
  result['recall_'+str(mm)+'mm']=float((errors<mm/1000).mean())
  result['random_'+str(mm)+'mm']=float((dist<mm/1000).mean())
  result['oracle_'+str(mm)+'mm']=float((dist.min(1)<mm/1000).mean())
  result['mutual_recall_'+str(mm)+'mm']=float((errors[mutual]<mm/1000).mean()) if mutual.any() else None
 # Internal diagnostic only: deterministic evenly sampled correspondence lines.
 vis=[]
 for im in d['images']:vis.append(cv2.cvtColor(cv2.resize(im,(448,448)),cv2.COLOR_RGB2BGR))
 canvas=np.concatenate(vis,axis=1)
 count=min(24,len(selected));chosen=np.linspace(0,len(selected)-1,count).astype(int)
 for j in chosen:
  s=d['source_grid'][d['source_indices'][j]];v=d['target_grid'][d['target_indices'][selected[j]]]
  a=np.asarray(d['source_box']);b=np.asarray(d['target_box'])
  x=np.rint((s-a[:2]+.5)/(a[2:]-a[:2])*448-.5).astype(int)
  y=np.rint((v-b[:2]+.5)/(b[2:]-b[:2])*448-.5).astype(int);y[0]+=448
  color=(0,210,255) if errors[j]<.015 else (180,100,220)
  cv2.line(canvas,tuple(x),tuple(y),color,1,cv2.LINE_AA)
 cv2.rectangle(canvas,(0,0),(896,35),(0,0,0),-1)
 cv2.putText(canvas,kind+' '+row['id']+' t'+str(t)+' geometry diagnostic',(6,23),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1,cv2.LINE_AA)
 image=RUN/(kind+'_'+row['id']+'_'+str(t)+'.jpg')
 assert not image.exists();cv2.imwrite(str(image),canvas,[cv2.IMWRITE_JPEG_QUALITY,88])
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['dinov2_s','tipsv2_b','dinov2_b']);p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
 RUN.mkdir(parents=True,exist_ok=True);m=manifest();save_json(RUN/'protocol.json',m)
 protocol_sha=hashlib.sha256((RUN/'protocol.json').read_bytes()).hexdigest()
 if args.prepare_only:print('protocol_locked',protocol_sha,flush=True);return
 torch.set_num_threads(4);torch.manual_seed(2026);net=model(args.model)
 for p in net.parameters():p.requires_grad_(False)
 print('model_ready',args.model,sum(p.numel() for p in net.parameters()),flush=True)
 out=RUN/(args.model+'.json')
 if out.exists():raise RuntimeError('complete output exists')
 rows=[]
 for row in m['episodes']:
  for t in row['frames']:
   rp=RUN/(args.model+'_'+row['id']+'_'+str(t)+'.json')
   if rp.exists():v=json.loads(rp.read_text());assert v['protocol_sha256']==protocol_sha
   else:
    v=evaluate(net,args.model,row,t);v['protocol_sha256']=protocol_sha;save_json(rp,v)
   rows.append(v);print(json.dumps(v),flush=True)
 save_json(out,dict(model=args.model,protocol_sha256=protocol_sha,rows=rows,
   caveat='RGB-D consistency on automatic candidate masks; not object identity truth or grasp accuracy; no downstream training'))
 print('complete',args.model,len(rows),flush=True)
if __name__=='__main__':main()

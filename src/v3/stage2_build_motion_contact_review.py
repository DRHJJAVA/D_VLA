from pathlib import Path
import json,cv2,numpy as np,h5py,hashlib,html
from stage2_probe_motion_fingerpads import project,corners
R=Path('/mnt/nvme1/drh/D_VLA')
D=R/'audits/v3/motion_finger_contact_pilot_v1'
V=R/'docs/图示/运动辅助夹指接触审核'
P=R/'docs/第二步_运动辅助夹指接触面审核.html'
def crop(im,center,rad=65,size=400):
 x,y=np.rint(center).astype(int);x0=max(0,x-rad);x1=min(im.shape[1],x+rad);y0=max(0,y-rad);y1=min(im.shape[0],y+rad)
 s=im[y0:y1,x0:x1]
 out=np.full((size,size,3),230,np.uint8);sc=size/max(s.shape[:2]);h,w=np.rint(np.array(s.shape[:2])*sc).astype(int);out[(size-h)//2:(size-h)//2+h,(size-w)//2:(size-w)//2+w]=cv2.resize(s,(w,h))
 return out
def main():
 V.mkdir(exist_ok=False);assert not P.exists()
 rows=json.loads((R/'audits/v3/finger_contact_repair_pilot_v1/protocol.json').read_text())['episodes']
 s=json.loads((D/'summary.json').read_text())
 by={x['id']:x for x in rows}
 choices=[('alphabet_soup_demo_0',1,60),('alphabet_soup_demo_1',1,56),('alphabet_soup_demo_1',2,90),('white_yellow_mug_demo_2',1,71),('tomato_sauce_demo_3',1,60),('tomato_sauce_demo_3',2,123)]
 cards=[];review=[]
 for ident,ai,t in choices:
  episode=next(x for x in s['episodes'] if x['id']==ident);attempt=episode['attempts'][ai-1];row=by[ident]
  accepted=t in attempt['accepted_contact_frames']
  with np.load(D/(ident+'.npz')) as z:
   offsets=z['frame_offsets'];start,end=offsets[t:t+2]
   points=z['points_world'][start:end];pixels=z['point_pixel'][start:end];cams=z['point_camera'][start:end];sides=z['point_side'][start:end];hand=z['hand_to_world'];q=z['measured_finger_q']
  if not accepted:
   points=points[:0];pixels=pixels[:0];cams=cams[:0];sides=sides[:0]
  counts=[int((sides==k).sum()) for k in [0,1]]
  name=ident+'_attempt'+str(ai)
  with h5py.File(row['rgbd'],'r') as f:
   main=[];zoom=[];context=[]
   for ci,cam in enumerate(['agentview','robot0_eye_in_hand']):
    g=f[cam];raw=cv2.cvtColor(g['rgb'][t],cv2.COLOR_RGB2BGR);mark=raw.copy()
    for u,v in pixels[cams==ci]:cv2.circle(mark,(int(u),int(v)),1,(0,220,0),-1)
    pads=np.concatenate([corners(q[t,k],side) for side,k in [(1,0),(-1,1)]])@hand[t,:3,:3].T+hand[t,:3,3]
    uv,zp=project(pads,g['K'][t],g['camera_to_world'][t])
    center=uv.mean(0)
    zoom.append(np.concatenate([crop(raw,center),crop(mark,center)],1))
    main.append(cv2.resize(raw,(400,400)))
   detail=np.concatenate(zoom,0)
   cv2.imwrite(str(V/(name+'_detail.jpg')),detail,[cv2.IMWRITE_JPEG_QUALITY,94])
   for ft in [max(0,attempt['start_frame']-2),t,attempt['end_frame']]:
    pair=[]
    for cam in ['agentview','robot0_eye_in_hand']:
     im=cv2.resize(cv2.cvtColor(f[cam]['rgb'][ft],cv2.COLOR_RGB2BGR),(320,320))
     cv2.rectangle(im,(0,0),(320,24),(0,0,0),-1);cv2.putText(im,'frame '+str(ft),(5,17),0,.5,(255,255,255),1);pair.append(im)
    context.append(np.concatenate(pair,1))
   cv2.imwrite(str(V/(name+'_context.jpg')),np.concatenate(context,0))
  if not accepted:
   status='空夹尝试：不输出成功接触面'
   note='该次闭合后夹指接近空夹位置，未取得物体随手抬升的视觉支持。图中不标成功接触面；下方保留实际动作画面。'
  elif all(counts):
   status='两侧均有可见深度支持，待你核对位置'
   note='绿色只标物体侧、紧邻两根夹指内侧垫面的实测RGB-D点；是接触位置估计，不是力传感器真值，也不是训练后模型评分。'
  else:
   status='持物证据成立，但只恢复了一侧接触位置'
   note='另一侧没有恢复出可见接触面，保留为未知；不以整块夹指投影、杯口或杯柄涂色补齐。此例尚未完成两侧接触面修复。'
  if ident=='tomato_sauce_demo_3' and ai==2:note+=' 完整序列显示第一次失败后还有这次抓取；此前整段失败的记录须改为按尝试审核，本次尚未获用户确认。'
  url='图示/运动辅助夹指接触审核/'+name
  card='<section><h2>'+ident+' · 第'+str(ai)+'次尝试</h2><p class="status">'+status+'</p>'
  card+='<p>闭合指令区间 '+str(attempt['start_frame'])+'–'+str(attempt['end_frame'])+' 帧；当前 '+str(t)+' 帧；夹持阶段后半段间距中位 '+f'{attempt["gap_late_median_mm"]:.1f}'+' mm。当前两侧可见窄带点数：'+str(counts[0])+' / '+str(counts[1])+'。</p><p>'+note+'</p>'
  card+='<div class="labels"><span>原始画面</span><span>接触位置估计（绿色）</span></div><p class="small">上排：主相机；下排：腕相机。这里只检查示范中的接触提取，不是评分模型的查询输入。</p><img src="'+url+'_detail.jpg" loading="lazy">'
  card+='<details><summary>展开：接近 → 当前时刻 → 本次闭合结束（双视角原图）</summary><img src="'+url+'_context.jpg" loading="lazy"></details></section>'
  cards.append(card)
  review.append({'id':ident,'attempt':ai,'frame':t,'accepted_by_motion_visual_gate':accepted,'visible_points_per_side':counts,'both_sides_visible':all(counts),'user_approved':False})
 css='body{font:17px/1.7 system-ui,sans-serif;max-width:1040px;margin:32px auto;padding:0 24px;background:#f4f6f8;color:#17232b}h1{font-size:30px}h2{font-size:23px}section{background:white;padding:24px;margin:24px 0;border:1px solid #d9e0e5;border-radius:10px}img{width:100%;height:auto}aside{background:#fff0c2;padding:18px}.labels{display:grid;grid-template-columns:1fr 1fr;text-align:center;font-weight:700}.status{font-weight:700}.small{font-size:14px;color:#4b5d69}summary{cursor:pointer;padding:12px}'
 page='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>运动辅助夹指接触面审核</title><style>'+css+'</style><body><h1>加入运动序列后的夹指接触面检查</h1><aside><b>尚未训练新模型，等待你审核。</b>4段示范、6次尝试分别处理。两次空夹不输出成功接触面；两例罐头两侧可见；杯壁与另一段罐体仍只恢复一侧，不能宣布全部修好。</aside><p>绿色 = 深度观测中位于夹指内侧0.3–2毫米窄带的物体侧点，已排除与机器人自身深度相符的像素。绿色不是评分高低，也不代表完整接触面积。基座、夹前触碰与未接触杯柄均不作为目标。</p><p>示范分析使用完整RGB-D、实际夹指关节和开合指令，以及已知机器人夹指尺寸与相机标定；没有使用物体仿真位姿或仿真接触。RGB-D帧t与官方运动观测t−1对齐。最终单物体评分输入不增加动作。</p>'+''.join(cards)+'</body></html>'
 P.write_text(page,encoding='utf-8')
 (D/'review_frames.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
 files=[p for p in D.iterdir() if p.is_file()]+[P]+list(V.glob('*.jpg'))+[R/'src/v3/stage2_motion_finger_contact_pilot.py',R/'src/v3/stage2_robot_only_depth.py',R/'src/v3/stage2_build_motion_contact_review.py']
 (D/'locked.json').write_text(json.dumps({'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
 print(json.dumps(review),flush=True)
if __name__=='__main__':main()

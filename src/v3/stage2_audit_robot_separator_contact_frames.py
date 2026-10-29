"""Visual-only contact-frame audit of auxiliary robot separator.
No simulator contacts, actions, names, human labels or held-out episodes are read.
Outputs internal remote gallery, not grasp-region prediction or Stage 2 review.
"""
from pathlib import Path
import json,sys
import cv2,h5py,numpy as np,torch
from PIL import Image,ImageDraw
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_train_preinteraction_robot_separator import SmallUNet,SIZE
S=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
SRC=R/'data/v3/samples/visual_interaction_uniform448_3dcc'
COMP=R/'data/v3/proposals/foreground3d_components'
MODEL=R/'outputs/v3_stage2/preinteraction_robot_seg_v1'
OUT=R/'audits/v3/stage2_robot_separator_contactframe_visual_audit.json'
FIG=R/'docs/图示/第二步_辅助机械臂分离器夹持帧内部核对.jpg'
def infer(net,rgb,depth,device):
 x=cv2.resize(rgb,(SIZE,SIZE),interpolation=cv2.INTER_AREA).astype(np.float32)/255
 z=cv2.resize(depth,(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
 x=np.concatenate([x,np.clip(z,0,2)[...,None]/2],axis=2)
 with torch.inference_mode():
  p=net(torch.from_numpy(x.transpose(2,0,1)[None]).to(device)).sigmoid()[0,0].cpu().numpy()
 return cv2.resize(p,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR)
def main():
 assert not OUT.exists() and not FIG.exists()
 summary=json.loads((MODEL/'summary.json').read_text())
 assert summary['status']=='auxiliary_robot_separator_pseudo_training_complete'
 ck=Path(summary['checkpoint'])
 device='cuda' if torch.cuda.is_available() else 'cpu'
 state=torch.load(ck,map_location=device,weights_only=False)
 net=SmallUNet().to(device);net.load_state_dict(state['model']);net.eval()
 ids=[('train','alphabet_soup_demo_3'),('train','porcelain_mug_demo_1'),
  ('train','tomato_sauce_demo_3'),('train','white_yellow_mug_demo_2')]
 ids += [('val',id) for id in S['val']]
 W=260;H=300;out=Image.new('RGB',(W*6,H*len(ids)),'white')
 rows=[]
 for i,(split,id) in enumerate(ids):
  with np.load(SRC/split/(id+'.npz')) as z:pix=z['pixel'].copy();event=int(z['onsets'][0])
  with np.load(COMP/split/(id+'.npz')) as z:
   path=Path(str(z['rgbd_path']));box=z['outer_box'].astype(int)
  x1,y1,x2,y2=box;x1=max(0,x1-85);y1=max(0,y1-85);x2=min(448,x2+85);y2=min(448,y2+85)
  frames=[0,event-1,min(event+4,9999)]
  with h5py.File(path) as f:
   frames[-1]=min(frames[-1],len(f['agentview/rgb'])-1)
   first=f['agentview/depth_m'][0]
   for j,t in enumerate(frames):
    rgb=f['agentview/rgb'][t];depth=f['agentview/depth_m'][t]
    prob=infer(net,rgb,depth,device);mask=prob>.5
    same=(depth[pix[:,1],pix[:,0]]>.05)&(np.abs(depth[pix[:,1],pix[:,0]]-first[pix[:,1],pix[:,0]])<.015)
    overlap=float(mask[pix[same,1],pix[same,0]].mean()) if same.any() else None
    row=dict(id=id,split=split,frame=t,relative=t-event,
     scene_robot_fraction=float(mask.mean()),
     still_visible_first_object_points=int(same.sum()),
     predicted_robot_on_still_visible_object=float(overlap) if overlap is not None else None)
    rows.append(row);print(json.dumps(row),flush=True)
    overlay=rgb.copy();overlay[mask]=(0.35*overlay[mask]+0.65*np.array([0,255,0])).astype(np.uint8)
    for k,(name,arr) in enumerate((('RGB',rgb),('predicted robot green',overlay))):
     a=arr[y1:y2,x1:x2]
     im=Image.fromarray(a);im.thumbnail((W-8,H-48))
     tile=Image.new('RGB',(W,H),'white')
     tile.paste(im,((W-im.width)//2,2))
     d=ImageDraw.Draw(tile);d.text((8,H-39),id,fill='black')
     d.text((8,H-24),f't={t} ({t-event:+d}) {name}',fill='black')
     d.text((8,H-9),f'visible-object overlap={overlap:.3f}' if overlap is not None else 'no visible object',fill='black')
     out.paste(tile,((j*2+k)*W,i*H))
 FIG.parent.mkdir(parents=True,exist_ok=True);out.save(FIG,quality=90)
 valid0=[x['predicted_robot_on_still_visible_object'] for x in rows if x['relative']<0 and x['frame']==0]
 validpre=[x['predicted_robot_on_still_visible_object'] for x in rows if x['relative']==-1]
 report=dict(status='auxiliary_robot_separator_visual_audit_not_contact_truth',
  model_checkpoint=str(ck),pseudo_val_loss=summary['best_pseudo_val_loss'],
  pseudo_val_iou=summary['best_pseudo_val_iou'],
  no_simulator_truth=True,no_actions=True,no_names=True,
  gallery=str(FIG),first_frame_object_leak_mean=float(np.mean(valid0)),
  pre_removal_still_visible_object_leak_mean=float(np.mean(validpre)),
  caution='mask may label occluding robot at object pixels; quantitative leakage tests only depth-still-visible first-frame object surface, not actual contact-frame object segmentation truth',
  rows=rows)
 OUT.write_text(json.dumps(report,indent=2))
 print(json.dumps(dict(first_frame_object_leak_mean=report['first_frame_object_leak_mean'],
  pre_removal_still_visible_object_leak_mean=report['pre_removal_still_visible_object_leak_mean'])),flush=True)
if __name__=='__main__':main()

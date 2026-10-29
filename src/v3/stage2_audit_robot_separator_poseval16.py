"""Pose-development visual-only audit of self-supervised robot separator.
Previously viewed same-asset 16 episodes. No simulator truth, actions, names,
manual labels or reserved blind data. The full query sequence locates an
audit frame only; this is not query-time grasp score inference.
"""
from pathlib import Path
import json,sys,collections
import cv2,h5py,numpy as np,torch
from PIL import Image,ImageDraw
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_train_preinteraction_robot_separator import SmallUNet
from stage2_audit_robot_separator_contact_frames import infer
MAN=R/'data/v3/manifests/stage2_pose_validation_16_rgbd.json'
SAMPLE=R/'data/v3/samples/object_only_dino_v1/poseval16'
TARGET=R/'audits/v3/stage2_poseval16_visual_targets'
MODEL=R/'outputs/v3_stage2/preinteraction_robot_seg_v1'
OUT=R/'audits/v3/stage2_robot_separator_poseval16_visual_audit.json'
FIG=R/'docs/图示/第二步_机械臂分离器姿态开发集内部核对.jpg'
def main():
 assert not OUT.exists() and not FIG.exists()
 entries=json.loads(MAN.read_text())['rows'];assert len(entries)==16
 s=json.loads((MODEL/'summary.json').read_text())
 device='cuda' if torch.cuda.is_available() else 'cpu'
 state=torch.load(s['checkpoint'],map_location=device,weights_only=False)
 net=SmallUNet().to(device);net.load_state_dict(state['model']);net.eval()
 rows=[];shown=[]
 for e in entries:
  id=e['id']
  with np.load(SAMPLE/(id+'.npz')) as z:pix=z['pixel'].copy();box=z['object_crop_box'].astype(int)
  with np.load(TARGET/(id+'.npz')) as z:event=int(z['onsets'][0])
  with h5py.File(e['rgbd_path']) as f:
   first=f['agentview/depth_m'][0]
   measurements=[]
   for t in (0,event-1):
    rgb=f['agentview/rgb'][t];depth=f['agentview/depth_m'][t]
    prob=infer(net,rgb,depth,device);mask=prob>.5
    still=(depth[pix[:,1],pix[:,0]]>.05)&(np.abs(depth[pix[:,1],pix[:,0]]-first[pix[:,1],pix[:,0]])<.015)
    overlap=float(mask[pix[still,1],pix[still,0]].mean()) if still.any() else None
    row=dict(id=id,asset_group=e['asset_group'],frame=t,relative=t-event,
     total_object_points=len(pix),still_visible_object_points=int(still.sum()),
     robot_predicted_on_still_visible_object=overlap,robot_mask_area_fraction=float(mask.mean()))
    rows.append(row);measurements.append(row);print(json.dumps(row),flush=True)
    if t==event-1 and (e['asset_group']=='white_yellow_mug' or e['source_episode'] in ('demo_13',)):
     shown.append((id,box,rgb,mask,overlap))
 valid=[x for x in rows if x['relative']==-1 and x['robot_predicted_on_still_visible_object'] is not None]
 summary=dict(episodes=len(entries),pre_event_mean_visible_object_overlap=float(np.mean([x['robot_predicted_on_still_visible_object'] for x in valid])),
  pre_event_max_visible_object_overlap=float(max(x['robot_predicted_on_still_visible_object'] for x in valid)),
  pre_event_mean_visible_object_fraction=float(np.mean([x['still_visible_object_points']/x['total_object_points'] for x in valid])))
 for g in sorted(set(x['asset_group'] for x in valid)):
  a=[x for x in valid if x['asset_group']==g]
  summary[g+'_pre_event_overlap_mean']=float(np.mean([x['robot_predicted_on_still_visible_object'] for x in a]))
 W=280;H=320;img=Image.new('RGB',(W*2,H*len(shown)),'white')
 for i,(id,box,rgb,mask,overlap) in enumerate(shown):
  x1,y1,x2,y2=box;x1=max(0,x1-80);y1=max(0,y1-80);x2=min(448,x2+80);y2=min(448,y2+80)
  overlay=rgb.copy();overlay[mask]=(0.35*overlay[mask]+0.65*np.array([0,255,0])).astype(np.uint8)
  for j,(label,a) in enumerate((('pre-lift RGB',rgb),('predicted robot green',overlay))):
   part=Image.fromarray(a[y1:y2,x1:x2]);part.thumbnail((W-8,H-48))
   tile=Image.new('RGB',(W,H),'white');tile.paste(part,((W-part.width)//2,2))
   d=ImageDraw.Draw(tile);d.text((7,H-36),id,fill='black');d.text((7,H-20),label,fill='black');d.text((7,H-8),f'visible-object overlap={overlap:.3f}',fill='black')
   img.paste(tile,(j*W,i*H))
 FIG.parent.mkdir(parents=True,exist_ok=True);img.save(FIG,quality=90)
 report=dict(status='poseval16_auxiliary_robot_visual_audit_only',
  no_simulator_truth=True,no_actions=True,no_names=True,
  previously_viewed_development=True,not_grasp_region_validation=True,
  gallery=str(FIG),summary=summary,rows=rows)
 OUT.write_text(json.dumps(report,indent=2))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

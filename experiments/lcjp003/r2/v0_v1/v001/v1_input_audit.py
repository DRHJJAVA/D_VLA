from pathlib import Path
import json,h5py,numpy as np,cv2
from PIL import Image,ImageDraw
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/lcjp003/r2/v0_v1/v001';A=R/'docs/LCJP003_R2_V01_assets'
d=json.loads((P/'v1_result.json').read_text()); records=[];tops=[]
for case in d['cases']:
 with h5py.File(R/'experiments/contact_joint_object/cache/observations'/f"{case['id']}.h5",'r') as ob,h5py.File(R/'experiments/contact_joint_object/cache/rgbd'/f"{case['id']}.h5",'r') as rgb:
  g=ob['frames'][str(case['frame'])]
  for v,cn in enumerate(['agentview','robot0_eye_in_hand']):
   raw=np.array(Image.open(A/case['images'][v]['raw']).resize((256,256)))
   canvas=raw.copy();robot=rgb[cn]['robot_mask'][case['frame']]>0
   labels=[]
   for c in case['candidates']:
    view=next((x for x in c['views'] if x['camera']==v),None)
    if view is None:continue
    mask=np.unpackbits(g[f"object_{c['id']}"][f'mask_{v}'][:]).reshape(256,256).astype(np.uint8)
    rr={'case':case['index'],'id':c['id'],'camera':v,'mask_area':int(mask.sum()),'robot_overlap_fraction':float((mask.astype(bool)&robot).sum()/max(mask.sum(),1)),'target_prior':view['target'],'is_aggregate_top':case['top_target']==c['id']}
    records.append(rr)
    if rr['is_aggregate_top']:tops.append(rr)
    color=(225,65,45) if rr['is_aggregate_top'] else (45,140,210)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(canvas,contours,-1,color,1)
    x0,y0,x1,y1=view['bbox'];labels.append((x0,y0,f"#{c['id']} {view['target']:.2f}",color))
   im=Image.fromarray(canvas).resize((512,512));dr=ImageDraw.Draw(im)
   for x,y,txt,color in labels:
    dr.rectangle((x*2,max(0,y*2-13),min(511,x*2+83),max(13,y*2)),fill='white');dr.text((x*2,max(0,y*2-13)),txt,fill=color)
   im.save(A/case['images'][v]['overlay'],quality=94)
summary={'cases':100,'candidate_views':len(records),'top_candidate_views':len(tops),'robot_overlap_fraction_quantiles_all':np.quantile([x['robot_overlap_fraction'] for x in records],[0,.5,.9,1]).tolist(),'robot_overlap_fraction_quantiles_top':np.quantile([x['robot_overlap_fraction'] for x in tops],[0,.5,.9,1]).tolist(),'descriptive_top_views_robot_overlap_over_half':sum(x['robot_overlap_fraction']>.5 for x in tops),'threshold_note':'0.5 only a descriptive summary of overlap; not used as object/contact decision or training target','manual_review_scope':[10,60],'manual_observations':[{'case':10,'observation':'bbq demo45 frame0 aggregate top #4 outlines robot/gripper rather than a task object; legacy #3 joins visually different appearances across cameras, requiring geometric identity recheck'},{'case':60,'observation':'milk demo45 frame0 visible red milk carton at left of external view lacks a separate candidate contour; top #3 instead outlines another cylindrical object. Large background mask does not count as an object proposal.'}],'v1_status':'input_quality_issues_found_not_passed','scope':'Frozen prior and old candidate cache diagnosis; not trained target-selection accuracy'}
(P/'v1_input_audit.json').write_text(json.dumps({'summary':summary,'records':records},ensure_ascii=False,indent=2))
(A/'v1_input_audit.json').write_text(json.dumps(summary,ensure_ascii=False))
print(json.dumps(summary,ensure_ascii=False))

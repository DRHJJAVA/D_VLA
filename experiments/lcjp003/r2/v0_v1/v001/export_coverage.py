from pathlib import Path
import json,hashlib,datetime,statistics
import numpy as np,cv2,h5py
from PIL import Image
R=Path('/mnt/nvme1/drh/D_VLA'); W=R/'experiments/lcjp003/r2/v0_v1/v001'; A=R/'docs/LCJP003_R2_V01_assets/coverage32'
def main():
 lock=json.loads((W/'v1_sampling_lock.json').read_text()); final=W/'segmentation_coverage.json'
 records=json.loads(final.read_text())['records'] if final.exists() else []
 lookup={(x['case'],x['camera'],x['grid']):x for x in records}
 reviewp=W/'segmentation_visual_review.json'
 reviews=json.loads(reviewp.read_text()) if reviewp.exists() else {'cases':{}}
 out=[]; count=0
 for ci,c in enumerate(lock['cases']):
  assert c['demo'] in ['demo_45','demo_46']
  row={'index':ci,'task':c['task'],'target':c['target'],'episode':c['demo'],'frame':c['frame'],'review':reviews['cases'].get(str(ci),{'status':'pending','note':'等待双视角逐例复核；不以候选数代替覆盖'}),'views':[]}
  with h5py.File(R/'experiments/contact_joint_object/cache/rgbd'/(c['id']+'.h5'),'r') as f:
   for v,cn in enumerate(['agentview','robot0_eye_in_hand']):
    rgb=f[cn]['rgb'][c['frame']]; robot=f[cn]['robot_mask'][c['frame']].astype(bool)
    raw=f'coverage_case{ci}_raw{v}.png'
    if not (A/raw).exists():Image.fromarray(rgb).save(A/raw)
    view={'camera':v,'raw':'LCJP003_R2_V01_assets/coverage32/'+raw,'raw_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'grids':{},'robot_mask_pixels':int(robot.sum())}
    for g in [8,32]:
     stem=f'coverage_case{ci}_view{v}_grid{g}'; p=W/(stem+'.npz')
     if not p.exists():continue
     with np.load(p) as z:masks=z['masks']
     rec=lookup.get((ci,v,g),{}); meta=rec.get('metadata',[]); proposals=[]
     for i,m in enumerate(masks):
      ys,xs=np.where(m);cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      props={'index':i,'original_proposal_index':meta[i]['index'] if i<len(meta) else None,'area':int(m.sum()),'bbox':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],'contours':[q[:,0,:].tolist() for q in cs],'robot_overlap':int((m&robot).sum())}
      dif=np.diff(np.concatenate(([0],m.reshape(-1).astype(np.int8),[0])))
      starts=np.flatnonzero(dif==1);ends=np.flatnonzero(dif==-1);props['rle']=np.stack([starts,ends-starts],axis=1).tolist()
      if i<len(meta):props.update({k:meta[i][k] for k in ['predicted_iou','stability_score']})
      proposals.append(props)
     view['grids'][str(g)]={'overlay':'LCJP003_R2_V01_assets/coverage32/'+stem+'.jpg','seconds':rec.get('seconds'),'proposals':proposals}
     count+=1
    row['views'].append(view)
  out.append(row)
 fusionp=W/'fusion_interface/result.json'
 fusion=json.loads(fusionp.read_text()) if fusionp.exists() else {'status':'pending','checks':[],'optimizer_updates':0}
 times={str(g):{'mean':statistics.mean([x['seconds'] for x in records if x['grid']==g]),'total':sum(x['seconds'] for x in records if x['grid']==g)} for g in [8,32]} if records else {}
 report={'generated_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),'summary':{'status':'completed_pending_review' if count==400 else 'running','completed_runs':count,'total_runs':400,'cases':100,'views':200,'seconds_by_grid':times,'reviewed_cases':sum(x['review']['status']!='pending' for x in out),'coverage_note':'SAM候选覆盖检查；冻结语义先验不作为目标真值。只对可见目标做候选复核，遮挡单列。原图/阈值/权重/后端相同，单变量8→32。','v1_passed':False,'optimizer_updates':0},'cases':out,'fusion':fusion}
 (A/'comparison.json').write_text(json.dumps(report,ensure_ascii=False))
 print(json.dumps({'exported_runs':count,'cases':len(out),'json_bytes':(A/'comparison.json').stat().st_size,'raw_frames':200,'source_frame_index':'sampling_lock frame'}))
if __name__=='__main__':
 try:main()
 except Exception as e:
  import traceback
  print(json.dumps({'type':type(e).__name__,'summary':str(e).splitlines()[0],'frames':[{'file':Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]}))

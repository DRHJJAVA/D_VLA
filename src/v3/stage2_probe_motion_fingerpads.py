from pathlib import Path
import json,numpy as np,h5py,cv2
from scipy.spatial.transform import Rotation
R=Path('/mnt/nvme1/drh/D_VLA')
O=R/'audits/v3/finger_contact_repair_pilot_v1'
V=R/'docs/图示/夹指接触修复检查'
def tc():
 t=np.eye(4);t[:3,:3]=Rotation.from_quat([.707108,.707108,0,0]).as_matrix()@np.diag([1,-1,-1]);t[:3,3]=[.05,0,0];return t
def project(p,K,T):
 c=(p-T[:3,3])@T[:3,:3];return np.stack([K[0,0]*c[:,0]/c[:,2]+K[0,2],K[1,1]*c[:,1]/c[:,2]+K[1,2]],1),c[:,2]
def cloud(d,K,T):
 v,u=np.indices(d.shape);p=np.stack([(u-K[0,2])*d/K[0,0],(v-K[1,2])*d/K[1,1],d],-1).reshape(-1,3)
 return p@T[:3,:3].T+T[:3,3]
def corners(q,side):
 y=q+side*.0035
 return np.array([[x,y+dy,z] for x in [-.008,.008] for dy in [-.004,.004] for z in [.0854,.1014]])
def main():
 rows=json.loads((O/'protocol.json').read_text())['episodes']
 E={x['id']:x for x in (json.loads(z) for z in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if z)}
 result=[]
 for row in rows:
  e=E[row['id']]
  with h5py.File(e['source_path'],'r') as f:
   g=f['data'][e['source_episode']];q=g['obs/gripper_states'][:];act=g['actions'][:,-1];ee=g['obs/ee_pos'][:]
  close=act>0;starts=np.flatnonzero(close&~np.r_[False,close[:-1]]);ends=np.flatnonzero(close&~np.r_[close[1:],False])
  qalign=np.concatenate([np.full((1,2),np.nan),q[:-1]],0)
  with h5py.File(row['rgbd'],'r') as f:
   hand=f['robot0_eye_in_hand/camera_to_world'][:]@np.linalg.inv(tc())
   episode={'id':row['id'],'split':row['split'],'attempts':[]}
   for idx,(a,b) in enumerate(zip(starts,ends)):
    ts=np.unique(np.r_[max(1,a-2),a+2,a+5,a+10,(a+b)//2,b]).clip(1,len(q)-1).astype(int)
    panels=[];stats=[]
    for t in ts:
     views=[]
     gaps=qalign[t,0]-qalign[t,1]-.001
     for cam in ['agentview','robot0_eye_in_hand']:
      c=f[cam];im=cv2.cvtColor(c['rgb'][t],cv2.COLOR_RGB2BGR)
      for side,k,col in [(1,0,(255,180,0)),(-1,1,(180,0,255))]:
       p=corners(qalign[t,k],side)@hand[t,:3,:3].T+hand[t,:3,3]
       uv,z=project(p,c['K'][t],c['camera_to_world'][t]);uv=np.rint(uv).astype(int)
       for i in range(8):
        for j in range(i+1,8):
         if (i^j) in [1,2,4] and z[i]>0 and z[j]>0:cv2.line(im,tuple(uv[i]),tuple(uv[j]),col,2)
      cv2.rectangle(im,(0,0),(im.shape[1],26),(0,0,0),-1)
      cv2.putText(im,f't={t} gap={gaps*1000:.1f}mm',(4,18),0,.5,(255,255,255),1)
      views.append(cv2.resize(im,(448,448)))
     panels.append(np.concatenate(views,1))
    jpg=V/(row['id']+'_attempt'+str(idx+1)+'_pads.jpg')
    cv2.imwrite(str(jpg),np.concatenate(panels,0))
    gap=qalign[a+1:b+1,0]-qalign[a+1:b+1,1]-.001
    entry={'attempt':idx+1,'close_command_start':int(a),'close_command_end':int(b),'gap_min_mm':float(np.nanmin(gap)*1000),'gap_late_median_mm':float(np.nanmedian(gap[-max(3,len(gap)//2):])*1000),'hand_z_rise_m':float(hand[a:b+1,2,3].max()-hand[a,2,3]),'shown_frames':ts.tolist()}
    episode['attempts'].append(entry)
   result.append(episode)
  print(episode,flush=True)
 (O/'motion_pad_probe.json').write_text(json.dumps({'alignment':'render frame t uses official measured gripper observation t-1; frame0 unknown','robot_geometry':'robosuite Panda finger pad boxes, no object/contact simulator truth','episodes':result},indent=2))
if __name__=='__main__':main()

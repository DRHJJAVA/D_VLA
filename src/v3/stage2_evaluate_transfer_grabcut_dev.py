"""Independent old-contact diagnostic of locked nonparametric visual transfer on development set."""
from pathlib import Path
import hashlib,json,math,sys
import h5py,numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_audit_preremoval_3d_visualcloud import project
M=R/'data/v3/manifests/stage2_localization_dev_8.json'
OUT=R/'outputs/v3_stage2/grabcut_dev_8'
TRANSFER=OUT/'transfer'
CONTACT=R/'data/v3/evaluation/localization_dev_contacts'
AUDIT=R/'audits/v3/stage2_transfer_grabcut_dev_contact_audit.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    lock=json.loads((TRANSFER/'transfer_locked.json').read_text())
    assert lock['status']=='train_visual_transfer_locked_before_contact_audit'
    entries=json.loads(M.read_text())['rows']
    results=[]
    for e in entries:
        id=e['id'];aid=e['asset_group'];ep=e['source_episode']
        path=TRANSFER/(id+'.npz');assert sha(path)==lock['prediction_sha256'][id]
        with np.load(path) as f:
            score=f['score'].copy();pts=f['xyz'].copy();box=f['source_box'].copy()
        with np.load(OUT/'predictions'/(id+'.npz')) as f:student=f['score'].copy()
        assert len(score)==len(student)==len(pts)
        state=json.loads((CONTACT/aid/(ep+'_states_contacts.json')).read_text())['rows']
        r0=state[0];p0=np.asarray(r0['object_position']);rot0=np.asarray(r0['object_rotation'])
        local=[]
        for row in state:
            p=np.asarray(row['object_position']);rot=np.asarray(row['object_rotation'])
            for c in row['contacts']:local.append((np.asarray(c['world'])-p)@rot)
        world=np.asarray(local)@rot0.T+p0
        with h5py.File(e['rgbd_path']) as f:
            depth=f['agentview/depth_m'][0];K=f['agentview/K'][0];T=f['agentview/camera_to_world'][0]
        u,v,visible=project(world,K,T,depth,.02)
        x1,y1,x2,y2=box;visible&=(u>=x1)&(u<x2)&(v>=y1)&(v<y2)
        cp=world[visible]
        if len(cp):
            near,ix=cKDTree(pts).query(cp);ix=ix[near<.015]
        else:ix=np.empty(0,int)
        row=dict(id=id,asset_group=aid,visual_points=len(pts),visible_contacts=len(cp),evaluable_contacts=len(ix))
        if len(ix):
            for fraction in (.1,.2,.3):
                k=max(1,math.ceil(fraction*len(score)))
                values=dict(random=k/len(score))
                for name,s in [('train_demo_transfer',score),('frozen_student',student)]:
                    top=np.argpartition(s,-k)[-k:]
                    values[name]=float(np.isin(ix,top).mean())
                row['budget_'+str(int(100*fraction))]=values
        results.append(row);print(json.dumps(row),flush=True)
    summary={}
    for budget in (10,20,30):
        rr=[x['budget_'+str(budget)] for x in results if 'budget_'+str(budget) in x]
        summary[str(budget)]={k:float(np.mean([q[k] for q in rr])) for k in ('train_demo_transfer','frozen_student','random')}
    summary['evaluable']=sum('budget_20' in x for x in results)
    assert not AUDIT.exists()
    AUDIT.write_text(json.dumps(dict(status='development_evaluation_only',old_simulator_contacts='unreviewed; no training',transfer_lock_sha256=sha(TRANSFER/'transfer_locked.json'),summary=summary,rows=results),indent=2))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

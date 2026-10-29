"""Frozen development-set contact diagnostic. Simulator references are opened only after prediction lock."""
from pathlib import Path
import hashlib,json,math,sys
import h5py,numpy as np
from scipy.spatial import cKDTree
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_audit_preremoval_3d_visualcloud import project
MANIFEST=R/'data/v3/manifests/stage2_localization_dev_8.json'
OUT=R/'outputs/v3_stage2/grabcut_dev_8'
LOCK=OUT/'predictions_locked.json'
CONTACT=R/'data/v3/evaluation/localization_dev_contacts'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def calc(e,lock):
    id=e['id'];path=OUT/'predictions'/(id+'.npz')
    assert sha(path)==lock['prediction_sha256'][id]
    with np.load(path) as f:
        xyz=f['xyz'].copy();pixel=f['pixel'].copy();box=f['source_box'].copy();score=f['score'].copy()
        assert int(f['query_frame'])==0 and not bool(f['has_future'])
    state=json.loads((CONTACT/e['asset_group']/(e['source_episode']+'_states_contacts.json')).read_text())
    r0=state['rows'][0];p0=np.asarray(r0['object_position']);rot0=np.asarray(r0['object_rotation'])
    local=[]
    for row in state['rows']:
        p=np.asarray(row['object_position']);rot=np.asarray(row['object_rotation'])
        for c in row['contacts']:local.append((np.asarray(c['world'])-p)@rot)
    base=dict(id=id,asset_group=e['asset_group'],object_points=len(xyz),reference_contacts_total=len(local))
    if not local:return dict(base,reason='no reference contacts')
    contacts=np.asarray(local)@rot0.T+p0
    with h5py.File(e['rgbd_path']) as f:
        depth=f['agentview/depth_m'][0];K=f['agentview/K'][0];T=f['agentview/camera_to_world'][0]
    u,v,visible=project(contacts,K,T,depth,.02)
    x1,y1,x2,y2=box
    visible&=(u>=x1)&(u<x2)&(v>=y1)&(v<y2)
    cp=contacts[visible]
    base['visible_reference_contacts']=len(cp)
    if not len(cp):return dict(base,reason='no visible reference contacts')
    nearest,ix=cKDTree(xyz).query(cp)
    ix=ix[nearest<.015]
    base['evaluable_reference_contacts']=len(ix)
    if not len(ix):return dict(base,reason='no near-cloud reference contacts')
    center3d=-np.linalg.norm(xyz-xyz.mean(0),axis=1)
    center2d=-np.linalg.norm(pixel-np.asarray([(x1+x2)/2,(y1+y2)/2]),axis=1)
    for frac in (.1,.2,.3):
        k=max(1,math.ceil(frac*len(xyz)))
        row={'random':k/len(xyz)}
        for name,s in [('student_ensemble',score),('center3d',center3d),('center2d',center2d)]:
            top=np.argpartition(s,-k)[-k:]
            row[name]=float(np.isin(ix,top).mean())
        base[f'budget_{int(frac*100)}']=row
    return base
def main():
    assert LOCK.exists(),'Predictions must be locked first'
    lock=json.loads(LOCK.read_text())
    assert lock['status']=='grabcut_development_predictions_locked_before_evaluation' and lock['no_query_future']
    assert sha(MANIFEST)==lock['manifest_sha256']
    rows_manifest=json.loads(MANIFEST.read_text())['rows']
    original=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
    oldids=set(sum(original.values(),[]))
    assert not (set(x['id'] for x in rows_manifest)&oldids)
    rows=[]
    for e in rows_manifest:
        row=calc(e,lock);rows.append(row);print(json.dumps(row),flush=True)
    summary={}
    for budget in (10,20,30):
        relevant=[r[f'budget_{budget}'] for r in rows if f'budget_{budget}' in r]
        summary[str(budget)]={k:float(np.mean([x[k] for x in relevant])) for k in ('student_ensemble','center3d','center2d','random')} if relevant else {}
    summary['evaluable_episodes']=sum('budget_20' in x for x in rows)
    summary['total_episodes']=len(rows)
    summary['visible_contact_episodes']=sum(x.get('visible_reference_contacts',0)>0 for x in rows)
    report=dict(status='provisional_evaluation_only',scope='development episodes demo_11/12 of same four training assets; first-frame DINO ROI plus fixed GrabCut object mask',prediction_lock_sha256=sha(LOCK),reference='old simulator contacts, unreviewed; measures only visible contact proximity, not gripper-enclosed region',development_set=True,summary=summary,rows=rows)
    target=OUT/'grabcut_dev_contact_diagnostic.json'
    assert not target.exists()
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

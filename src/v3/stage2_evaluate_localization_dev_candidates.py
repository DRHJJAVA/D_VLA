"""Independent contact-projection audit of frozen development localization candidates.
Contacts are unreviewed simulator references for evaluation only, never rank training."""
from pathlib import Path
import json,sys
import h5py,numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
from stage2_audit_preremoval_3d_visualcloud import project
M=R/'data/v3/manifests/stage2_localization_dev_8.json'
LOCK=R/'audits/v3/stage2_localization_dev_8_candidates_locked.json'
OUT=R/'audits/v3/stage2_localization_dev_8_contact_audit.json'
CONTACT=R/'data/v3/evaluation/localization_dev_contacts'
def box_score(box,u,v,visible):
    x1,y1,x2,y2=box
    inside=(u>=x1)&(u<x2)&(v>=y1)&(v<y2)
    return dict(box=box,area_fraction=(x2-x1)*(y2-y1)/(448*448),visible_contact_coverage=float((visible&inside).sum()/visible.sum()) if visible.any() else None,projected_contact_coverage=float(inside.mean()) if len(inside) else None)
def main():
    assert json.loads(LOCK.read_text())['status']=='candidate_scores_frozen_before_simulator_contact_evaluation'
    entries=json.loads(M.read_text())['rows']
    original_dino={x['id']:x for x in json.loads((R/'audits/v3/stage2_dino_template_dev_8.json').read_text())['cases']}
    masked_dino={x['id']:x for x in json.loads((R/'audits/v3/stage2_dino_masked_template_dev_8.json').read_text())['cases']}
    results=[]
    for e in entries:
        id=e['id'];aid=e['asset_group'];ep=e['source_episode']
        state=json.loads((CONTACT/aid/(ep+'_states_contacts.json')).read_text())
        r0=state['rows'][0];p0=np.asarray(r0['object_position']);rot0=np.asarray(r0['object_rotation'])
        local=[]
        for row in state['rows']:
            p=np.asarray(row['object_position']);rot=np.asarray(row['object_rotation'])
            for c in row['contacts']:local.append((np.asarray(c['world'])-p)@rot)
        if not local:
            results.append(dict(id=id,reason='no reference contacts'));continue
        world=np.asarray(local)@rot0.T+p0
        with h5py.File(e['rgbd_path']) as f:
            depth=f['agentview/depth_m'][0];K=f['agentview/K'][0];T=f['agentview/camera_to_world'][0]
        u,v,visible=project(world,K,T,depth,.02)
        raw=json.loads((R/'data/v3/proposals/first_frame_templates/dev'/(id+'.json')).read_text())['candidates']
        masked=json.loads((R/'data/v3/proposals/first_frame_masked_templates/dev'/(id+'.json')).read_text())['candidates']
        d1=original_dino[id]['rows'];d2=masked_dino[id]['rows']
        j1=int(np.argmax([q['dino_cls_max'] for q in d1]));j2=int(np.argmax([q['dino_cls_max'] for q in d2]))
        fullseq=json.loads((R/'data/v3/proposals/motion_rgbd_rerank'/aid/(ep+'.json')).read_text())['selected']['box']
        boxes=dict(raw_template=raw[0]['box'],masked_template=masked[0]['box'],raw_dino=d1[j1]['box'],masked_dino=d2[j2]['box'],full_sequence_visual=fullseq)
        row=dict(id=id,reference_contacts=len(world),depth_visible_contacts=int(visible.sum()),methods={k:box_score(b,u,v,visible) for k,b in boxes.items()})
        results.append(row)
        print(json.dumps(row),flush=True)
    summary={}
    for key in ('raw_template','masked_template','raw_dino','masked_dino','full_sequence_visual'):
        rr=[q['methods'][key] for q in results if 'methods' in q and q['methods'][key]['visible_contact_coverage'] is not None]
        summary[key]=dict(evaluable=len(rr),mean_visible_contact_coverage=float(np.mean([x['visible_contact_coverage'] for x in rr])),episodes_ge_05=sum(x['visible_contact_coverage']>=.5 for x in rr),mean_area_fraction=float(np.mean([x['area_fraction'] for x in rr])))
    report=dict(status='evaluation_only',old_contacts='unreviewed simulator contacts; never training or candidate ranking',frozen_candidate_lock=str(LOCK),summary=summary,rows=results)
    assert not OUT.exists();OUT.write_text(json.dumps(report,indent=2))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

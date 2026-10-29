import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','4')
import sys,pathlib,json,time,traceback
R=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/r3/joint/v001')
sys.path.insert(0,str(R/'src'))
from supervision import prepare_episode,supervision
from geometry import policy_input,OLD
def main():
    ep=json.loads((R/'episodes.json').read_text())[0]
    t0=time.time();result=prepare_episode(ep)
    frames=sorted(int(p.stem) for p in (OLD/'cache'/ep['id']).glob('*.pt') if (p.parent/('observations_'+p.stem+'.h5')).exists())
    selected=[frames[int((len(frames)-1)*f)] for f in [.1,.3,.5,.7,.9]]
    records=[]
    for t in selected:
        d,y,objects,meta=policy_input(ep,t)
        s=supervision(ep,t,objects,d)
        idx=int(s['role'].argmax())
        records.append({'frame':t,'objects':len(objects),'target_index':idx,'target_candidate_ids':objects[idx]['ids'] if idx<len(objects) else [],'target_component':objects[idx].get('component',0) if idx<len(objects) else None,'target_probability':float(s['role'].max()),'supervision_weight':s['role_weight'],'target_evidence':s['valid_target_evidence'],'part_peak':float(s['parts'].max()),'crossview_pairs':meta['crossview_pairs']})
    (R/'checks/supervision_pilot.json').write_text(json.dumps({'episode':ep['id'],'tracking':result,'windows':records,'seconds':time.time()-t0,'visual_review_passed':False},indent=2))
    print(json.dumps({'pilot_complete':True,'seconds':time.time()-t0}))
if __name__=='__main__':
    try:main()
    except Exception as e:
        result={'type':type(e).__name__,'message':str(e)[:300],'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]}
        (R/'checks/pilot_failure.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));sys.exit(1)

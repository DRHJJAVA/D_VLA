
import sys,pathlib,json,ast,time,traceback
import numpy as np,torch
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object')
sys.path[:0]=[str(B/'src'),str(B/'engineering')]
import data,input_cache,network
from online_targets import target_for
def main():
    torch.set_num_threads(4)
    ck=torch.load(B/'runs/v001/last.pt',map_location='cpu',weights_only=False)
    m=network.JointModel().cuda();m.load_state_dict(ck['model']);m.eval()
    ds=data.Dataset('train')
    tree=ast.parse((B/'src/network.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='joint_loss')
    terms=['total','motion','depth','reproj','geom','pen','complexity','visual','nll','scoreloss','consistency']
    fn.body[-1]=ast.Return(value=ast.Dict(keys=[ast.Constant(k) for k in terms],values=[ast.Name(id=k,ctx=ast.Load()) for k in terms]))
    scope=dict(vars(network));exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<loss_diagnostic>','exec'),scope)
    loss_terms=scope['joint_loss'];rows=[]
    for r in ds.rows[::20]:
        for t in np.linspace(1,r['frames']-2,8,dtype=int):
            for oid in ds.frame_objects(r,int(t)):
                raw=input_cache.cached_get(ds,r,int(t),oid,ci=int(t)%2,point_limit=128)
                if len(raw['motion_source'])<3 or np.min(np.abs(raw['sdf']))>3*ds.sigma:continue
                b=data.to_device(raw);online=target_for(raw,3)
                torch.manual_seed(789)
                with torch.no_grad():out=m(b,True)
                class Fixed:
                    def __init__(self,c):self.c=c
                    def __call__(self,*a,**kw):
                        z=dict(out);z['contact']=self.c;return z
                result={'id':r['id'],'frame':int(t),'object':int(oid),'current_contact_mean':float(out['contact'].mean()),
                    'current_score_mean':float(out['score'].mean()),'current_score_std':float(out['score'].std()),
                    'current_interval_width':float((out['upper']-out['lower']).mean()),
                    'candidates':[]}
                for value in [0.,1e-5,.01,.1,.5]:
                    c=torch.full_like(out['contact'],value,requires_grad=True)
                    values=loss_terms(Fixed(c),b,online)
                    entry={'contact_value':value,**{k:float(v.detach()) for k,v in values.items()}}
                    g=torch.autograd.grad(values['total'],c)[0]
                    entry['mean_dloss_dcontact']=float(g.mean())
                    result['candidates'].append(entry)
                rows.append(result)
                break
            if len(rows)>=24:break
        if len(rows)>=24:break
    summary={'checkpoint_epoch':ck['epoch'],'samples':len(rows),'selection':'first deterministic train samples within 3 observation sigma of either finger and >=3 tracked points; not contact truth',
        'zero_contact_lowest_among_sweep':sum(min(r['candidates'],key=lambda x:x['total'])['contact_value']==0 for r in rows),
        'positive_derivative_at_zero':sum(r['candidates'][0]['mean_dloss_dcontact']>0 for r in rows),
        'contact_mean':float(np.mean([r['current_contact_mean'] for r in rows])),
        'score_mean':float(np.mean([r['current_score_mean'] for r in rows])),
        'mean_within_proposal_score_std':float(np.mean([r['current_score_std'] for r in rows])),
        'interval_width':float(np.mean([r['current_interval_width'] for r in rows]))}
    (B/'engineering/collapse_diagnostic.json').write_text(json.dumps({'summary':summary,'rows':rows},indent=2))
    print(json.dumps(summary))
if __name__=='__main__':
    try:main()
    except Exception as e:
        (B/'engineering/collapse_failure.json').write_text(json.dumps({'error_type':type(e).__name__,'locations':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]}))
        with (B/'engineering/collapse_traceback.log').open('w') as f:traceback.print_exc(file=f)

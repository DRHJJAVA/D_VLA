
import sys,pathlib,json,time,copy,random,traceback
import numpy as np,torch
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object')
sys.path[:0]=[str(B/'src'),str(B/'engineering')]
import data,input_cache
from network import JointModel,joint_loss
from online_targets import target_for
def run():
    torch.set_num_threads(4)
    ck=torch.load(B/'runs/v001/last.pt',map_location='cpu',weights_only=False)
    cfg=ck['config'];train=data.Dataset('train');dev=data.Dataset('dev')
    schedule=json.loads((B/'runs/v001/schedule.json').read_text())['windows']
    plan=[x for x in schedule if x['epoch']==2][:8]
    tree=__import__('ast').parse((B/'src/train.py').read_text())
    # Copy exact object-selection assignment from the original training loop.
    ast=__import__('ast')
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and n.lineno==111)
    selector=compile(ast.Module(body=[node],type_ignores=[]),'<original_oid_selector>','exec')
    samples=[]
    for i,item in enumerate(plan):
        r=train.rows[item['row_index']];t=item['frame'];ids=train.frame_objects(r,t)
        scope={'ids':ids,'epoch':2,'item':item,'i':i,'j':i,'r':r,'t':t}
        # Discover loop variable rather than assume its name.
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and n.lineno==108)
        if isinstance(loop.target,ast.Tuple):scope[loop.target.elts[0].id]=i
        exec(selector,scope);oid=scope['oid']
        samples.append((train,r,t,oid,(2+item['quarter'])%2,2048))
    for ri in [0,15,30,49]:
        r=dev.rows[ri]
        for t in [0,min(30,r['frames']-1),r['frames']//2,r['frames']-1]:
            for oid in dev.frame_objects(r,t)[:2]:samples.append((dev,r,t,oid,t%2,128))
    equal=True;baseline=[];cached=[];cold=[]
    original_inputs=[]
    for ds,r,t,o,c,n in samples:
        start=time.perf_counter();a=input_cache.ORIGINAL_GET(ds,r,t,o,c,n);baseline.append(time.perf_counter()-start)
        start=time.perf_counter();b=input_cache.cached_get(ds,r,t,o,c,n);cold.append(time.perf_counter()-start)
        start=time.perf_counter();craw=input_cache.cached_get(ds,r,t,o,c,n);cached.append(time.perf_counter()-start)
        equal=equal and input_cache.equivalent(a,b) and input_cache.equivalent(a,craw)
        original_inputs.append(a)
    assert equal,'raw_input_mismatch'
    def rng(seed=12345):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    def train_once(cache):
        rng()
        m=JointModel().cuda();m.load_state_dict(ck['model']);m.train()
        opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        opt.load_state_dict(copy.deepcopy(ck['optimizer']))
        # Parameter construction must not consume the RNG used by the paired run.
        rng(34567);opt.zero_grad(set_to_none=True);losses=[];outputs=[];active=[]
        for ix,(ds,r,t,o,c,n) in enumerate(samples[:8]):
            raw=input_cache.cached_get(ds,r,t,o,c,n) if cache else input_cache.ORIGINAL_GET(ds,r,t,o,c,n)
            b=data.to_device(raw);online=target_for(raw,2)
            loss,v,out=joint_loss(m,b,online);(loss/8).backward()
            losses.append(float(loss.detach()));active.append(float(v['active_score_points']))
            outputs.append({k:x.detach().cpu() for k,x in out.items() if torch.is_tensor(x)})
        grads={k:p.grad.detach().cpu().clone() for k,p in m.named_parameters() if p.grad is not None}
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        states={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        return losses,active,outputs,grads,states
    a=train_once(False);b=train_once(True)
    def maxdiff(x,y):
        if isinstance(x,dict):return max([maxdiff(x[k],y[k]) for k in x] or [0.])
        if isinstance(x,(list,tuple)):return max([maxdiff(v,w) for v,w in zip(x,y)] or [0.])
        if torch.is_tensor(x):return 0. if torch.equal(x,y) else float((x.to(torch.float64)-y.to(torch.float64)).abs().max()) if x.numel() else 0.
        return abs(float(x)-float(y))
    diff={'loss':maxdiff(a[0],b[0]),'outputs':maxdiff(a[2],b[2]),'gradients':maxdiff(a[3],b[3]),'after_adamw_update':maxdiff(a[4],b[4])}
    assert all(v==0 for v in diff.values()),'numerical_mismatch'
    m=JointModel().cuda();m.load_state_dict(ck['model']);m.eval()
    timings={}
    for cache in [False,True]:
        rng(56789);torch.cuda.synchronize();start=time.perf_counter()
        with torch.no_grad():
            for ds,r,t,o,c,n in samples[8:]:
                raw=input_cache.cached_get(ds,r,t,o,c,n) if cache else input_cache.ORIGINAL_GET(ds,r,t,o,c,n)
                joint_loss(m,data.to_device(raw))
        torch.cuda.synchronize();timings['cached' if cache else 'original']=time.perf_counter()-start
    result={'checkpoint_epoch':ck['epoch'],'samples':len(samples),'raw_exact':equal,'differences':diff,
        'active_points_in_optimizer_test':sum(a[1]),'input_original_seconds':sum(baseline),
        'input_cold_seconds':sum(cold),'input_warm_seconds':sum(cached),
        'development_sample_seconds':timings,'development_sample_speedup':timings['original']/timings['cached'],
        'cache_token':input_cache.TOKEN,'time':time.time()}
    (B/'engineering/verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=='__main__':
    try:run()
    except Exception as e:
        result={'error_type':type(e).__name__,'locations':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]}
        (B/'engineering/verification_failure.json').write_text(json.dumps(result));print(json.dumps(result))

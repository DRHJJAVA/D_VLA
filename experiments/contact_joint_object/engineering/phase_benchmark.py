
import pathlib,sys,ast,json,time,traceback,copy,random
import torch,numpy as np,h5py
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object')
sys.path[:0]=[str(B/'src'),str(B/'engineering')]
import data,input_cache,train,online_targets
def compare_h5(a,b):
    bad=[]
    with h5py.File(a,'r') as x,h5py.File(b,'r') as y:
        def visit(name,obj):
            if isinstance(obj,h5py.Dataset):
                if name not in y or not np.array_equal(obj[()],y[name][()],equal_nan=True):bad.append(name)
        x.visititems(visit)
    return bad
def main():
    torch.set_num_threads(4)
    ck=torch.load(B/'runs/v001/last.pt',map_location='cpu',weights_only=False)
    teacher=train.JointModel().cuda();teacher.load_state_dict(ck['ema']);teacher.eval()
    dtrain=data.Dataset('train');dev=data.Dataset('dev')
    row=next(r for r in dtrain.rows if 'bbq_sauce' in r['id'] and r['episode']=='demo_0')
    drow=next(r for r in dev.rows if 'bbq_sauce' in r['id'] and r['episode']=='demo_45')
    dev.rows=[drow]
    root=B/'engineering/phase_benchmark';root.mkdir(exist_ok=True)
    tree=ast.parse((B/'src/online_targets.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='refresh_episode')
    assign=next(n for n in fn.body if n.lineno==14)
    assign.value=ast.parse("destination",mode='eval').body
    scope=dict(vars(online_targets))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<engineering_refresh_benchmark>','exec'),scope)
    # Suppress development report writes and training-state updates in the benchmark subprocess.
    et=ast.parse((B/'src/train.py').read_text());ef=next(n for n in et.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    es=dict(vars(train));es['save']=lambda *a,**k:None;es['state']=lambda *a,**k:None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ef],type_ignores=[])),'<engineering_development_benchmark>','exec'),es)
    results={}
    for mode in ['original','cached']:
        data.Dataset.get=input_cache.ORIGINAL_GET if mode=='original' else input_cache.cached_get
        torch.manual_seed(2026);np.random.seed(2026);random.seed(2026)
        scope['destination']=root/(mode+'.h5')
        torch.cuda.synchronize();start=time.time()
        online=scope['refresh_episode'](teacher,dtrain,row,-3)
        torch.cuda.synchronize();online_seconds=time.time()-start
        torch.manual_seed(2026);np.random.seed(2026);random.seed(2026)
        model=train.JointModel().cuda();model.load_state_dict(ck['model']);model.eval()
        start=time.time();metric=es['evaluate'](model,dev,-3);torch.cuda.synchronize();dev_seconds=time.time()-start
        results[mode]={'online_episode_seconds':online_seconds,'development_episode_seconds':dev_seconds,'development_criterion':metric}
        del model;torch.cuda.empty_cache()
    bad=compare_h5(root/'original.h5',root/'cached.h5')
    (B/'engineering/phase_comparison_details.json').write_text(json.dumps({'results':results,'bad_arrays':bad}))
    assert not bad and results['original']['development_criterion']['criterion']==results['cached']['development_criterion']['criterion'] and results['original']['development_criterion']['episodes']==results['cached']['development_criterion']['episodes'],'phase_result_mismatch'
    result={'checkpoint_epoch':ck['epoch'],'train_case':row['id'],'dev_case':drow['id'],'results':results,'online_arrays_exact':True,'development_metric_exact':True,
        'online_speedup':results['original']['online_episode_seconds']/results['cached']['online_episode_seconds'],
        'development_speedup':results['original']['development_episode_seconds']/results['cached']['development_episode_seconds'],
        'full_epoch_estimate_minutes_from_prior_phase_durations':65/(results['original']['online_episode_seconds']/results['cached']['online_episode_seconds'])+45/(results['original']['development_episode_seconds']/results['cached']['development_episode_seconds'])+3,
        'cache_stats':input_cache.STATS,'time':time.time()}
    (B/'engineering/phase_benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    try:main()
    except Exception as e:
        (B/'engineering/phase_benchmark_failure.json').write_text(json.dumps({'error_type':type(e).__name__,'locations':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]}))
        with (B/'engineering/phase_benchmark_traceback.log').open('w') as f:traceback.print_exc(file=f)

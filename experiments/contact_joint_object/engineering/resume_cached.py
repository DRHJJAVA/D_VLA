
"""Resume the unchanged original epoch loop, with immutable input caching only."""
import pathlib,sys,ast,json,time,hashlib,os,traceback,shutil,subprocess
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object')
P=B.parents[1]
sys.path[:0]=[str(B/'src'),str(B/'engineering')]
import train as original
import input_cache
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def logtime(stage,elapsed,**extra):
    with (B/'engineering/runtime_timings.jsonl').open('a') as f:
        f.write(json.dumps({'time':time.time(),'stage':stage,'seconds':elapsed,**extra})+'\n')
def preview_snapshot(epoch):
    if epoch!=5:return
    dest=B/'runs/v001/preview_epoch005.pt'
    if not dest.exists():
        tmp=dest.with_suffix('.tmp');shutil.copyfile(B/'runs/v001/last.pt',tmp);tmp.replace(dest)
    manifest={'epoch':epoch,'checkpoint_sha256':sha(dest),'split':'development','case':'pick_up_the_bbq_sauce_and_place_it_in_the_basket__demo_45','training_continues':True,'not_final_heldout_test':True,'preview_script_sha256':sha(B/'engineering/preview_video.py'),'time':time.time()}
    original.save(B/'engineering/preview_request.json',manifest)
    script=B/'engineering/preview_video.py'
    if script.exists():
        log=open(B/'engineering/preview.log','a')
        p=subprocess.Popen([sys.executable,str(script)],stdout=log,stderr=log,start_new_session=True)
        manifest['pid']=p.pid;original.save(B/'engineering/preview_request.json',manifest)
def main():
    verification=json.loads((B/'engineering/verification.json').read_text())
    assert verification['raw_exact'] and all(v==0 for v in verification['differences'].values())
    assert input_cache.TOKEN==verification['cache_token']
    input_cache.install()
    old_evaluate=original.evaluate
    old_refresh=original.refresh_episode
    def evaluate(*a,**kw):
        start=time.time();result=old_evaluate(*a,**kw)
        logtime('development',time.time()-start,epoch=a[2],cache_stats=dict(input_cache.STATS));return result
    def refresh(*a,**kw):
        start=time.time();result=old_refresh(*a,**kw)
        logtime('online_episode',time.time()-start,epoch=a[3],episode=a[2]['id'],cache_stats=dict(input_cache.STATS));return result
    original.evaluate=evaluate;original.refresh_episode=refresh
    tree=ast.parse((B/'src/train.py').read_text())
    func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in func.body if isinstance(n,ast.For) and n.lineno==94)
    original_loop=ast.dump(loop,include_attributes=False)
    loop.iter.args[0]=ast.parse("checkpoint['epoch']+1",mode='eval').body
    loop.body.append(ast.parse('preview_snapshot(epoch)').body[0])
    prefix=[n for n in func.body if n.lineno in (54,55)]
    prefix+=ast.parse("""
run=B/'runs/v001'
checkpoint=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
assert checkpoint['epoch']>=3
assert checkpoint['config']==cfg
inputs=json.loads((run/'input_locked.json').read_text())['files']
sigma=checkpoint['measurement_sigma']
train.sigma=sigma
dev.sigma=sigma
""").body
    prefix +=[n for n in func.body if n.lineno in (75,76,77)]
    prefix+=ast.parse("""
m.load_state_dict(checkpoint['model'])
ema.load_state_dict(checkpoint['ema'])
opt.load_state_dict(checkpoint['optimizer'])
source_hash=json.loads((run/'source_locked.json').read_text())['files']
for rel,h in source_hash.items():
    assert sha(P/rel)==h, 'original_source_hash_mismatch'
source_hash=dict(source_hash)
for p in [B/'engineering/input_cache.py',B/'engineering/resume_cached.py']:
    source_hash[str(p.relative_to(P))]=sha(p)
save(run/'source_engineering_locked.json',{'time':now(),'files':source_hash,'inherits':'source_locked.json','resume_epoch':checkpoint['epoch'],'resume_updates':checkpoint['optimizer_updates'],'checkpoint_sha256':sha(run/'last.pt'),'network_loss_split_evaluation_unchanged':True})
assert sha(run/'schedule.json')==checkpoint['schedule_sha256']
sampler=json.loads((run/'schedule.json').read_text())['windows']
history=json.loads((run/'history.json').read_text())
assert len(history)==checkpoint['epoch'] and history[-1]['optimizer_updates']==checkpoint['optimizer_updates']
best=json.loads((run/'best_selection.json').read_text())['criterion']
updates=checkpoint['optimizer_updates']
torch.set_rng_state(checkpoint['torch_rng'])
torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
np.random.set_state(checkpoint['numpy_rng'])
random.setstate(checkpoint['python_rng'])
assert torch.equal(torch.get_rng_state(),checkpoint['torch_rng'])
assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),checkpoint['cuda_rng']))
assert random.getstate()==checkpoint['python_rng']
assert np.array_equal(np.random.get_state()[1],checkpoint['numpy_rng'][1])
assert all(torch.equal(v.cpu(),checkpoint['model'][k]) for k,v in m.state_dict().items())
assert all(torch.equal(v.cpu(),checkpoint['ema'][k]) for k,v in ema.state_dict().items())
loaded_optimizer=opt.state_dict()
assert loaded_optimizer['param_groups']==checkpoint['optimizer']['param_groups']
assert all(torch.equal(v.cpu(),checkpoint['optimizer']['state'][k][name].cpu()) if torch.is_tensor(v) else v==checkpoint['optimizer']['state'][k][name] for k,state_values in loaded_optimizer['state'].items() for name,v in state_values.items())

save(B/'engineering/resumed.json',{'time':now(),'epoch':checkpoint['epoch'],'optimizer_updates':updates,'rng_restored':True,'optimizer_restored':True,'ema_restored':True,'cache_token':input_cache.TOKEN,'pid':os.getpid()})
""").body
    func.body=prefix+[loop]+[n for n in func.body if n.lineno>=139]
    # Confirm the actual training/evaluation epoch loop has only the boundary and preview hook edits.
    check=__import__('copy').deepcopy(loop);check.iter.args[0]=ast.Constant(1);check.body.pop()
    assert ast.dump(check,include_attributes=False)==original_loop,'epoch_loop_changed'
    module=ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[]))
    scope=dict(vars(original));scope.update(preview_snapshot=preview_snapshot,input_cache=input_cache)
    exec(compile(module,'<unchanged_loop_with_resume>','exec'),scope)
    scope['main']()
if __name__=='__main__':
    try:main()
    except Exception as e:
        detail={'error_type':type(e).__name__,'locations':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)],'time':time.time()}
        original.save(B/'engineering/resume_failure.json',detail)
        with (B/'engineering/resume_traceback.log').open('w') as f:traceback.print_exc(file=f)
        sys.exit(1)

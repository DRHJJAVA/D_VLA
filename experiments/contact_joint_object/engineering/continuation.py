
import os,sys,pathlib,json,time,signal,subprocess,hashlib,traceback,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
B=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object');P=B.parents[1]
PY='/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python'
def save(name,obj):
    p=B/'engineering'/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def command(pid):
    p=pathlib.Path('/proc')/str(pid)/'cmdline'
    return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''
def main():
    import fcntl
    handle=open(B/'engineering/continuation.lock','w')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify=json.loads((B/'engineering/verification.json').read_text())
    assert verify['raw_exact'] and all(x==0 for x in verify['differences'].values())
    assert json.loads((B/'engineering/resume_validation.json').read_text())['passed']
    assert (B/'engineering/preview_video.py').exists(),'preview_not_ready'
    parent,worker=2814455,2814456
    assert str(B/'src/pipeline.py') in command(parent)
    assert str(B/'src/train.py') in command(worker)
    save('continuation_state.json',{'stage':'waiting_for_complete_epoch_boundary','time':now(),'pid':os.getpid(),'original_parent':parent,'original_worker':worker})
    while True:
        if not command(worker):raise RuntimeError('original_worker_exited_before_handoff')
        state=json.loads((B/'training_state.json').read_text())
        history=json.loads((B/'runs/v001/history.json').read_text())
        if (B/'engineering/phase_benchmark.json').exists() and history and history[-1]['epoch']>=3 and state.get('stage')=='online_target_refresh' and state.get('epoch')==history[-1]['epoch']+1:
            os.kill(worker,signal.SIGSTOP)
            # Recheck after stopping: never discard a partially updated next epoch.
            state=json.loads((B/'training_state.json').read_text())
            if state.get('stage')!='online_target_refresh' or state.get('epoch')!=history[-1]['epoch']+1:
                os.kill(worker,signal.SIGCONT);time.sleep(5);continue
            import torch
            ck=torch.load(B/'runs/v001/last.pt',map_location='cpu',weights_only=False)
            if ck['epoch']!=history[-1]['epoch'] or ck['optimizer_updates']!=history[-1]['optimizer_updates']:
                os.kill(worker,signal.SIGCONT);raise RuntimeError('checkpoint_history_mismatch')
            assert str(B/'src/pipeline.py') in command(parent)
            backup=B/'runs/v001'/('pre_engineering_epoch%03d.pt'%ck['epoch'])
            assert not backup.exists()
            shutil.copyfile(B/'runs/v001/last.pt',backup)
            assert hashlib.sha256(backup.read_bytes()).digest()==hashlib.sha256((B/'runs/v001/last.pt').read_bytes()).digest()
            os.kill(parent,signal.SIGTERM)
            os.kill(worker,signal.SIGTERM);os.kill(worker,signal.SIGCONT)
            save('handoff.json',{'time':now(),'epoch':ck['epoch'],'optimizer_updates':ck['optimizer_updates'],'last_sha256':hashlib.sha256((B/'runs/v001/last.pt').read_bytes()).hexdigest(),'next_online_progress_at_stop':state,'no_optimizer_updates_discarded':True})
            with (P/'docs/无接触标签联合学习_LIBERO-Object网络与训练验证方案.md').open('a') as f:
                f.write('\n\n### '+now()+' E4检查点接续\n完整保存第'+str(ck['epoch'])+'轮、累计'+str(ck['optimizer_updates'])+'次更新后，在下一轮参数更新前接续缓存版。原始源保持不变，完整恢复model/EMA/optimizer及全部RNG，沿用原schedule、输入锁、best准则；没有丢弃已完成的参数更新。工程继承记录在runs/v001/source_engineering_locked.json。\n')
            break
        time.sleep(5)
    log=open(B/'engineering/resume.log','a')
    child=subprocess.Popen([PY,str(B/'engineering/resume_cached.py')],stdout=log,stderr=log,start_new_session=True)
    save('continuation_state.json',{'stage':'resumed_training','time':now(),'pid':os.getpid(),'training_pid':child.pid})
    rc=child.wait()
    if rc!=0:raise RuntimeError('resumed_worker_failed')
    assert (B/'runs/v001/frozen_for_test.json').exists()
    log=open(B/'engineering/final_test.log','a')
    test=subprocess.Popen([PY,str(B/'src/evaluate_frozen.py')],stdout=log,stderr=log,start_new_session=True)
    save('continuation_state.json',{'stage':'frozen_final_test','time':now(),'pid':os.getpid(),'test_pid':test.pid})
    rc=test.wait()
    if rc!=0:raise RuntimeError('frozen_final_test_failed')
    save('continuation_state.json',{'stage':'complete_pending_visual_review','time':now()})
if __name__=='__main__':
    try:main()
    except Exception as e:
        if str(B/'src/train.py') in command(2814456):
            try:os.kill(2814456,signal.SIGCONT)
            except ProcessLookupError:pass
        save('continuation_failure.json',{'error_type':type(e).__name__,'message':str(e)[:160],'time':now(),'locations':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})
        with (B/'engineering/continuation_traceback.log').open('w') as f:traceback.print_exc(file=f)

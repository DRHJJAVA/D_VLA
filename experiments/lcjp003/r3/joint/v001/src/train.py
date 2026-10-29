import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,pathlib,json,time,random,hashlib,traceback,fcntl
import numpy as np,torch
from concurrent.futures import ThreadPoolExecutor
R=pathlib.Path('/mnt/nvme1/drh/D_VLA');RUN=R/'experiments/lcjp003/r3/joint/v001'
sys.path.insert(0,str(RUN/'src'))
from geometry import policy_input,OLD
from network import JointNetwork
from objective import targets,loss
CFG=json.loads((RUN/'config.json').read_text())
EP=json.loads((RUN/'episodes.json').read_text());BYID={e['id']:e for e in EP}
def save(name,value):
    p=RUN/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(p)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def state(status,step,**extra):save('state.json',dict(status=status,completed_updates=step,time=time.time(),pid=os.getpid(),**extra))
def load_window(e,t):
    d,y,o,m=policy_input(e,t);return d,targets(e,t,d,y,o)
def device(d):return {k:v.cuda() if torch.is_tensor(v) else v for k,v in d.items()}
def checkpoint(model,opt,step):
    d={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'step':step,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},'source_lock_sha256':sha(RUN/'source_lock.json'),'config_sha256':sha(RUN/'config.json')}
    p=RUN/'checkpoints/last.pt';tmp=p.with_suffix('.tmp');torch.save(d,tmp);tmp.replace(p)
    if step in [0,20,50] or step%250==0:
        import shutil
        shutil.copy2(p,RUN/'checkpoints'/('step%04d.pt'%step))
def evaluate(model,step):
    model.eval();records=[];t0=time.time()
    windows=json.loads((OLD/'eval0000.json').read_text())['records']
    with torch.no_grad():
        for i,w in enumerate(windows):
            ep=BYID[w['episode']];d,y,objects,meta=policy_input(ep,w['frame']);d=device(d)
            gen=torch.Generator(device='cpu').manual_seed(20261028+i);z=torch.randn(1,16,6,generator=gen).cuda()
            enc=model.encode(d);a,g=model.generate(enc,z)
            target=y['target'].cuda();err=(a[0]-target).abs()
            nr=json.loads((RUN/'normalization.json').read_text());scale=torch.tensor(nr['std'][:6],device='cuda')
            rec={'episode':w['episode'],'split':w['split'],'frame':w['frame'],'normalized_motion_mae':float(err.mean()),'translation_action_mae':float((err*scale)[:,:3].mean()),'rotation_action_mae':float((err*scale)[:,3:].mean()),'gripper_error':float((g[0].argmax(-1).cpu()!=y['grip']).float().mean()),'target_index':int(enc['roles'].argmax()),'target_confidence':float(enc['roles'].softmax(-1).max()),'lambda_mean':float(enc['gate'].mean()),'score_point_std_mean':float(enc['alpha'].std(-1).mean())}
            for intervention in ['uniform','shuffle']:
                ei=model.encode(d,intervention);ai,_=model.generate(ei,z)
                rec[intervention+'_mae']=float((ai[0]-target).abs().mean())
            chosen=int(enc['roles'].argmax())
            rec['selected_candidate_ids']=objects[chosen]['ids'] if chosen<len(objects) else []
            rec['selected_component']=objects[chosen].get('component',0) if chosen<len(objects) else None
            directory=RUN/'eval_assets'/('step%04d'%step);directory.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(directory/(w['episode']+'_'+str(w['frame'])+'.npz'),xyz=d['xyz'].cpu().numpy(),valid=d['valid'].cpu().numpy(),alpha=enc['alpha'].cpu().numpy(),roles=enc['roles'].softmax(-1).cpu().numpy(),selected=chosen,actions=a[0].cpu().numpy(),lambda_values=enc['gate'].cpu().numpy())
            records.append(rec)
    summary={}
    for split in sorted(set(v['split'] for v in records)):
        selected=[v for v in records if v['split']==split]
        summary[split]={k:float(np.mean([v[k] for v in selected])) for k in ['normalized_motion_mae','translation_action_mae','rotation_action_mae','gripper_error','uniform_mae','shuffle_mae']}
    save('eval%04d.json'%step,{'step':step,'seconds':time.time()-t0,'same_80_windows':True,'same_noise':True,'summary':summary,'records':records,'P1_or_method_passed':False})
    model.train();return time.time()-t0
def main():
    lock=(RUN/'train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for gate in ['network_interface','visibility_unit','drift_reference','supervision_review','resume_equivalence','crossview_geometry','input_completeness','joint_objective']:
        if not json.loads((RUN/'checks'/(gate+'.json')).read_text()).get('passed',False):raise RuntimeError('gate_not_passed:'+gate)
    expected=json.loads((RUN/'source_lock.json').read_text())['files']
    assert all(sha(R/p)==v for p,v in expected.items()),'source_changed_after_lock'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(CFG['seed']);np.random.seed(CFG['seed']);torch.manual_seed(CFG['seed']);torch.cuda.manual_seed_all(CFG['seed'])
    sys.path.insert(0,str(OLD/'src'));import common
    official=common.build_expert('cpu');expert=official.lm_expert;del official
    model=JointNetwork(expert).cuda()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    start=0;p=RUN/'checkpoints/last.pt'
    if p.exists():
        ck=torch.load(p,map_location='cpu',weights_only=False)
        assert ck['config_sha256']==sha(RUN/'config.json') and ck['source_lock_sha256']==sha(RUN/'source_lock.json')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['step']
        rs=ck['rng'];random.setstate(rs['python']);np.random.set_state(rs['numpy']);torch.set_rng_state(rs['torch']);torch.cuda.set_rng_state_all(rs['cuda'])
    else:checkpoint(model,opt,0)
    evaluation_seconds=evaluate(model,0) if not (RUN/'eval0000.json').exists() else json.loads((RUN/'eval0000.json').read_text())['seconds']
    train=[e for e in EP if e['v2_split']=='train']
    frames={e['id']:sorted(int(p.stem) for p in (OLD/'cache'/e['id']).glob('*.pt')) for e in train}
    tasks=sorted(set(e['task'] for e in train));history=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for step in range(start+1,CFG['updates']+1):
            if (RUN/'STOP').exists():
                checkpoint(model,opt,step-1);state('paused_by_request',step-1);return
            t0=time.time();picked=[]
            for _ in range(8):
                task=random.choice(tasks);e=random.choice([e for e in train if e['task']==task]);t=random.choice(frames[e['id']]);picked.append((e,t))
            pending=[pool.submit(load_window,e,t) for e,t in picked]
            opt.zero_grad(set_to_none=True);metrics=[];timing={'data_h2d':0.,'forward':0.,'backward':0.,'optimizer':0.}
            for j,future in enumerate(pending):
                tt=time.time();d,y=future.result();d=device(d);y=device(y);torch.cuda.synchronize();timing['data_h2d']+=time.time()-tt
                tt=time.time();z=torch.randn(8,16,6,device='cuda');value,info=loss(model,d,y,z)
                # Paired causal window consistency, not matching equal array indices.
                e,t=picked[j]
                if j==0 and t+1 in frames[e['id']]:
                    dn,_,_,_=policy_input(e,t+1);dn=device(dn)
                    en=model.encode(d);a0,g0=model.generate(en,z[:1])
                    plan=torch.cat([a0[0,1:].detach(),(g0[0,1:].argmax(-1).float()*2-1)[:,None]],-1)
                    plan=torch.cat([plan,plan[-1:]],0)
                    if random.random()<.5:plan=None
                    a1,_=model.generate(model.encode(dn),z[:1].roll(-1,1),plan=plan)
                    continuity=(a1[0,:15]-a0[0,1:].detach()).square().mean()
                    value=value+8*.02*continuity;info['continuity']=float(continuity.detach())
                torch.cuda.synchronize();timing['forward']+=time.time()-tt
                if not torch.isfinite(value):raise FloatingPointError('nonfinite_joint_loss')
                tt=time.time();(value/8).backward();torch.cuda.synchronize();timing['backward']+=time.time()-tt;metrics.append(info)
            gradient={}
            for name in ['role','score','gate']:
                gradient[name]=float(sum(p.grad.square().sum() for p in getattr(model,name).parameters() if p.grad is not None).sqrt())
                if not np.isfinite(gradient[name]):raise FloatingPointError('nonfinite_gradient:'+name)
            tt=time.time();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.cuda.synchronize();timing['optimizer']=time.time()-tt
            elapsed=time.time()-t0;history.append(elapsed)
            with (RUN/'runtime_timings.jsonl').open('a') as f:f.write(json.dumps({'update':step,'seconds':elapsed,'timings':timing,'gradients':gradient,'windows':[{'episode':e['id'],'frame':t} for e,t in picked],'metrics':metrics})+'\n')
            state('training',step,last_update_seconds=elapsed)
            if step%10==0 or step in [20,50]:checkpoint(model,opt,step)
            if step in [20,50]:
                stable=history[-min(len(history),30):]
                save('eta%04d.json'%step,{'step':step,'mean_update_seconds':float(np.mean(stable)),'median_update_seconds':float(np.median(stable)),'to_step250_seconds':max(0,250-step)*np.mean(stable)+evaluation_seconds,'to_step3000_seconds':(3000-step)*np.mean(stable)+(3000-step)//250*evaluation_seconds,'shared_gpu_load_may_change':True,'cold_preparation_excluded':True})
            if step%250==0:
                state('evaluating',step);evaluate(model,step);checkpoint(model,opt,step)
    state('training_complete_pending_acceptance',3000,automatic_closed_loop=False)
if __name__=='__main__':
    try:main()
    except Exception as e:
        save('failure.json',{'type':type(e).__name__,'summary':str(e)[:300],'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]})
        sys.exit(1)

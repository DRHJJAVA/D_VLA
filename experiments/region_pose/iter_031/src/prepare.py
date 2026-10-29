from pathlib import Path
import ast,copy,json,hashlib,shutil,time
R=Path('/mnt/nvme1/drh/D_VLA');B=R/'experiments/region_pose/iter_031';O=R/'experiments/region_pose/iter_029'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
assert not (B/'src/train.py').exists()
oldlock=json.loads((O/'completed_locked.json').read_text())
cfg=json.loads((O/'config.json').read_text());cfg['iteration']='031';cfg['epochs']=24;cfg['scope']='029 expanded10 fixed240; training rank known-negative top64 only';cfg['runs']=[r for r in cfg['runs'] if r['group']=='expanded10'];cfg['rank_negative_selection']='stable_descending_current_detached_logit_top64_known_only';cfg['validation_loss']='original029';cfg['source_lock_sha256']=sha(O/'source_locked.json')
dump(B/'config.json',cfg)
for name in ['model.py','pointnet2_grasp.py']:
 shutil.copyfile(O/'src'/name,B/'src'/name)
for name in ['manifest.json']:
 shutil.copyfile(O/name,B/name)
for name in ['samples','graphs']:(B/name).symlink_to(O/name,target_is_directory=True)
for p in (O/'initialization').iterdir():
 if p.is_file():shutil.copyfile(p,B/'initialization'/p.name)
for run in cfg['runs']:
 n=run['group']+'_'+run['variant'];shutil.copyfile(O/'plans'/f'{n}.json',B/'plans'/f'{n}.json')
tree=ast.parse((O/'src/train.py').read_text())
original_loss=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='loss')
candidate=copy.deepcopy(original_loss);candidate.name='training_loss'
count=0
for n in ast.walk(candidate):
 if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ix' and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='linspace' for x in ast.walk(n.value)):
  n.value=ast.parse('ix[torch.argsort(logit[ix].detach(), descending=True, stable=True)[:min(64, len(ix))]]',mode='eval').body;count+=1
assert count==1
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
calls=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='loss' and n.lineno==169]
assert len(calls)==3
for n in calls:n.func.id='training_loss'
tree.body.insert(tree.body.index(original_loss)+1,candidate)
tree=ast.fix_missing_locations(tree)
(B/'src/train.py').write_text(ast.unparse(tree)+'\n')
new=ast.parse((B/'src/train.py').read_text())
same={}
old=ast.parse((O/'src/train.py').read_text())
for name in ['loadrows','forward','loss','metrics','evaluate','atomicjson']:
 a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
 b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
 same[name]=ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False);assert same[name]
back=copy.deepcopy(new);back.body=[n for n in back.body if not(isinstance(n,ast.FunctionDef) and n.name=='training_loss')]
for n in ast.walk(back):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='training_loss':n.func.id='loss'
assert ast.dump(back,include_attributes=False)==ast.dump(old,include_attributes=False)
dump(B/'audits/method_identity.json',{'unchanged_functions':same,'only_candidate_function_and_three_training_calls':True,'model_identical':sha(B/'src/model.py')==sha(O/'src/model.py'),'pointnet_identical':sha(B/'src/pointnet2_grasp.py')==sha(O/'src/pointnet2_grasp.py')})
files=[O/'completed_locked.json',O/'manifest.json',O/'config.json',O/'src/train.py',O/'src/model.py',O/'src/pointnet2_grasp.py',O/'source_locked.json',O/'evaluation/baselines.json']
files+=list((O/'samples').glob('*.npz'))+list((O/'graphs').glob('*.npz'))+list((O/'initialization').glob('*'))
m=json.loads((B/'manifest.json').read_text())
for run in cfg['runs']:
 name=run['group']+'_'+run['variant']
 files += [O/'plans'/f'{name}.json',O/'runs'/name/'best.pt',O/'runs'/name/'last.pt',O/'runs'/name/'history.json']
 for snap in ['best','last']:
  files += [O/'evaluation'/name/snap/f"{row['key']}.npz" for row in m['rows']]
locked={}
for p in files:
 if not p.is_file():continue
 k=str(p.relative_to(R));h=sha(p)
 if p.name!='completed_locked.json':assert oldlock[k]==h,k
 locked[k]=h
dump(B/'input_locked.json',locked)
dump(B/'stage_state.json',{'stage':'prepared_pending_preflight','time':time.time()})
print(json.dumps({'prepared':True,'inputs':len(locked),'samples':len(m['rows']),'runs':len(cfg['runs'])}))

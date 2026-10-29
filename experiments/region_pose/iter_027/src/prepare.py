from pathlib import Path
import ast,json,hashlib,shutil
R=Path('/mnt/nvme1/drh/D_VLA');P=R/'experiments/region_pose';B=P/'iter_027'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
src=P/'iter_014/src/diagnose_early_queries.py'
t=ast.parse(src.read_text())
class Rewrite(ast.NodeTransformer):
 def visit_Constant(self,n):
  if isinstance(n.value,str):
   if n.value=='iter_013/targets/train/summary.json':return ast.copy_location(ast.Constant('iter_027/new_cases.json'),n)
   if n.value=='train':return ast.copy_location(ast.Constant('raw_new'),n)
   if n.value in ['train/summary.json','train/locked.json']:return ast.copy_location(ast.Constant(n.value.replace('train/','raw_new/')),n)
   if n.value=='stage_state.json':return ast.copy_location(ast.Constant('new_tracking_state.json'),n)
  return n
t=Rewrite().visit(t);ast.fix_missing_locations(t)
(B/'src/track_new.py').write_text(ast.unparse(t)+'\n')
shutil.copyfile(P/'iter_014/src/tracking_math.py',B/'src/tracking_math.py')
(B/'raw_new').mkdir()
rows=json.loads((P/'iter_025/targets/train/summary.json').read_text())['rows']
eligible={x['episode'] for x in json.loads((P/'iter_025/sampling/train/summary.json').read_text())['rows'] if x['eligible']}
old=json.loads((P/'iter_014/train/summary.json').read_text())['rows']
oldeps={x['episode'] for x in old};new=eligible-oldeps
assert len(eligible)==10 and len(new)==4
dump(B/'cases.json',{'rows':[x for x in rows if x['episode'] in eligible]})
dump(B/'new_cases.json',{'rows':[x for x in rows if x['episode'] in new]})
inputs=set(json.loads((P/'iter_014/input_locked.json').read_text()))
inputs.update(json.loads((P/'iter_014/completed_locked.json').read_text()))
for f in ['iter_025/targets/train/summary.json','iter_025/sampling/train/summary.json','iter_025/completed_locked.json']:
 inputs.add(str((P/f).relative_to(R)))
protocol={x['id']:x for x in json.loads((P/'iter_003/evidence/base/protocol.json').read_text())['episodes']}
for c in rows:
 if c['episode'] not in eligible:continue
 ident=c['episode'];inputs.add(str(Path(c['target_source']).relative_to(R)))
 for ext in ['npz','json']:inputs.add(str((P/'iter_003/evidence/motion'/f'{ident}.{ext}').relative_to(R)))
 inputs.add(str(Path(protocol[ident]['component']).relative_to(R)))
dump(B/'input_locked.json',{f:sha(R/f) for f in sorted(inputs)})
source=[B/'设计方案.md',B/'config.json',B/'cases.json',B/'new_cases.json',B/'src/track_new.py',B/'src/tracking_math.py',src,P/'iter_025/src/sampling.py',R/'src/v3/stage2_probe_motion_fingerpads.py',R/'src/v3/stage2_robot_only_depth.py',B/'src/preflight.py']
dump(B/'source_locked.json',{str(f.relative_to(R)):sha(f) for f in source})
dump(B/'stage_state.json',{'stage':'preflight_pending','new_events':sorted(new),'reuse_events':sorted(oldeps)})
print(json.dumps({'new_events':sorted(new),'reuse_events':len(oldeps),'inputs':len(inputs),'sources':len(source)}))

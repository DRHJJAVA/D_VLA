from pathlib import Path
import json,hashlib,sys,runpy,traceback
R=Path('/mnt/nvme1/drh/D_VLA/experiments/region_pose/iter_035')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8388608),b''):h.update(x)
 return h.hexdigest()
try:
 for name in ['source_locked.json','input_locked.json','seed_locked.json']:
  for p,h in json.loads((R/name).read_text()).items():assert sha(p)==h,'pre-forward hash mismatch'
 assert json.loads((R/'audits/preflight.json').read_text())['tracking_allowed']
 (R/'stage_state.json').write_text(json.dumps({'stage':'tracking'}))
 sys.path.insert(0,str(R/'src'))
 sys.argv=[str(R/'src/track.py'),'train']
 runpy.run_path(str(R/'src/track.py'),run_name='__main__')
 (R/'stage_state.json').write_text(json.dumps({'stage':'tracking_finished'}))
except Exception as e:
 (R/'track_traceback.txt').write_text(traceback.format_exc())
 (R/'track_failure.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e)}))
 (R/'stage_state.json').write_text(json.dumps({'stage':'tracking_failed'}))
 print('tracking failed; details retained remotely')
 raise SystemExit(1)

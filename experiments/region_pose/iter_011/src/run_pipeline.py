from pathlib import Path
import json,sys,subprocess,hashlib,fcntl,time
P=Path(__file__).resolve().parents[1]
handle=(P/'pipeline.lock').open('a')
fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
lock=json.loads((P/'source_locked.json').read_text())
assert all(hashlib.sha256((P/f).read_bytes()).hexdigest()==h for f,h in lock.items())
for split in ['train','val']:
 summary=P/split/'summary.json'
 (P/'stage_state.json').write_text(json.dumps({'stage':split,'updated':time.time()}))
 if not summary.exists():
  with (P/split/'run.log').open('a') as log:
   ret=subprocess.run([sys.executable,'-u',str(P/'src/extend_cotracker.py'),split],stdout=log,stderr=subprocess.STDOUT)
  if ret.returncode:
   (P/'blocked.json').write_text(json.dumps({'stage':split,'returncode':ret.returncode}))
   raise SystemExit(ret.returncode)
 slock=json.loads((P/split/'locked.json').read_text())
 assert all(hashlib.sha256((P/split/f).read_bytes()).hexdigest()==h for f,h in slock.items())
 if split=='train':
  freeze={'train_summary_sha256':hashlib.sha256(summary.read_bytes()).hexdigest(),'source_locked_sha256':hashlib.sha256((P/'source_locked.json').read_bytes()).hexdigest(),'rules_changed':False}
  dest=P/'pre_validation_freeze.json'
  if dest.exists():assert json.loads(dest.read_text())==freeze
  else:dest.write_text(json.dumps(freeze,indent=2))
(P/'stage_state.json').write_text(json.dumps({'stage':'complete','updated':time.time()}))
print('PIPELINE_COMPLETE',flush=True)

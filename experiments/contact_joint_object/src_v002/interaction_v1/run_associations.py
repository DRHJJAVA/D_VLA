import pathlib,json,subprocess,time,datetime
e=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object');r=e/'runs/v002/preflight/P1_interaction_v1'
p=json.loads((r/'protocol.json').read_text())
for ep in p['episodes']:
 while not (r/'segmentation'/f'{ep}.h5').exists():
  if (r/'segmentation_failure.json').exists():raise SystemExit(2)
  time.sleep(5)
 with (r/(ep+'_association.log')).open('w') as out:
  done=subprocess.run(['/mnt/nvme1/drh/miniforge3/envs/D_VLA/bin/python',str(e/'src_v002/interaction_v1/associate_render.py'),'--episode',ep],stdout=out,stderr=subprocess.STDOUT)
 if done.returncode:raise SystemExit(done.returncode)
(r/'association_all_complete.json').write_text(json.dumps({'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds'),'episodes':p['episodes']}))

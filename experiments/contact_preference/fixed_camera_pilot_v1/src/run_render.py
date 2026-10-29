from pathlib import Path
import json,subprocess,sys,os,datetime
root=Path('/mnt/nvme1/drh/D_VLA');out=root/'experiments/contact_preference/fixed_camera_pilot_v1'
cfg=json.loads((out/'config.json').read_text())
for name,c in cfg['cases'].items():
    (out/'stage_state.json').write_text(json.dumps({'stage':'rendering','case':name,'time':datetime.datetime.now().isoformat()}))
    args=[sys.executable,str(out/'src/render_fixed_sparse.py'),'--input',str(root/c['raw']),'--output',str(out/(name+'.h5')),'--libero-root','/home/drh/project/ECV-VLA/vendor/LIBERO','--bddl-subdir',c['suite'],'--episode','demo_0','--size','448','--frames',','.join(map(str,c['frames']))]
    with (out/(name+'_render.log')).open('w') as h:
        result=subprocess.run(args,stdout=h,stderr=subprocess.STDOUT,env={**os.environ,'MUJOCO_GL':'egl'})
    if result.returncode:
        (out/'stage_state.json').write_text(json.dumps({'stage':'render_failed','case':name,'returncode':result.returncode}))
        sys.exit(result.returncode)
(out/'stage_state.json').write_text(json.dumps({'stage':'render_complete'}))

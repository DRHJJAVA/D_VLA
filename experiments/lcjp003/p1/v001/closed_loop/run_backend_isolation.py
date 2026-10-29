import subprocess, pathlib
c=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003/p1/v001/closed_loop')
with (c/'probe_sam_math_tf32_probe.py.log').open("w") as f: subprocess.run(['/mnt/nvme1/drh/D_VLA/experiments/lcjp003/runtime/venv/bin/python', str(c/'probe_sam_math_tf32_probe.py')],cwd=c,stdout=f,stderr=f)
with (c/'probe_sam_default_fp32_probe.py.log').open("w") as f: subprocess.run(['/mnt/nvme1/drh/D_VLA/experiments/lcjp003/runtime/venv/bin/python', str(c/'probe_sam_default_fp32_probe.py')],cwd=c,stdout=f,stderr=f)

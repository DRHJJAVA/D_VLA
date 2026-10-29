from common13 import *
import subprocess,traceback,time
def main():
 stage('starting_training')
 log=(BASE/'training.log').open('w')
 p=subprocess.Popen([sys.executable,str(BASE/'src/train13.py')],cwd=BASE,stdout=log,stderr=log)
 save(BASE/'runner_child.json',dict(pid=p.pid,time=now()))
 code=p.wait();log.close()
 if code:
  lines=(BASE/'training.log').read_text(errors='replace').splitlines()
  locations=[x.strip() for x in lines if x.startswith('  File ')]
  error=[x for x in lines if x.startswith(('AssertionError:','RuntimeError:','TypeError:','ValueError:','KeyError:','IndexError:','NameError:','AttributeError:'))]
  save(BASE/'failure.json',dict(time=now(),returncode=code,locations=locations,error=error[-1:]));stage('training_failed',returncode=code)
 else:stage('training_complete_pending_audit')
if __name__=='__main__':main()

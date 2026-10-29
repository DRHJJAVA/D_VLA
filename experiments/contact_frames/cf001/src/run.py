from common import *
import subprocess,sys
def main():
 for name in ["prepare","train","verify","finalize"]:
  with (RUN/(name+".log")).open("w") as f:
   p=subprocess.Popen([sys.executable,str(RUN/"src"/(name+".py"))],cwd=ROOT,stdout=f,stderr=f)
   stage("running_"+name,child_pid=p.pid)
   try:ret=p.wait(timeout=3600)
   except subprocess.TimeoutExpired:
    p.terminate();p.wait(timeout=30);stage("timeout",step=name);return
  if ret:stage("failed",step=name,returncode=ret);return
if __name__=="__main__":
 try:main()
 except Exception as e:fail(e)

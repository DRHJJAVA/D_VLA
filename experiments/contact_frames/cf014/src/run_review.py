from common14 import *
import subprocess,sys,traceback
def main():
    assert (BASE/'weights_frozen.json').exists()
    from prepare_features import main as prepare
    prepare('review')
    subprocess.run([sys.executable,'-u',str(BASE/'src/evaluate14.py')],check=True,cwd=BASE)
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(BASE/'failure_review_runner.json',dict(time=now(),type=type(e).__name__,message=str(e),
          stack=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_tb(e.__traceback__)]))
        stage('review_runner_failed');raise

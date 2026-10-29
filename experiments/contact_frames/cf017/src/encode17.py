from common14 import *
import h5py,numpy as np,subprocess,imageio_ffmpeg
def main():
 verify(json.loads((BASE/'predictions_frozen.json').read_text())['files'])
 assets=ROOT/'docs/图示/cf017_contact'
 for c in CFG['cases']:
  ep=c['id'];out=assets/f'{ep}_clean.mp4';conf=json.loads(episode_config(ep).read_text())
  assert not out.exists()
  with h5py.File(conf['rgbd'],'r') as h,open(BASE/'cases'/ep/'encode_fixed.log','w') as log:
   a=h['agentview/rgb'];b=h['robot0_eye_in_hand/rgb'];H,W=a.shape[1:3]
   cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W*2}x{H}','-r','20','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
   proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
   for i in range(len(a)):proc.stdin.write(np.concatenate([a[i],b[i]],axis=1).tobytes())
   proc.stdin.close();assert proc.wait()==0
 (ROOT/'docs/CF017_跨任务接触预测查看.html').write_bytes((BASE/'src/review17.html').read_bytes())
 stage('prediction_page_ready')
if __name__=='__main__':main()

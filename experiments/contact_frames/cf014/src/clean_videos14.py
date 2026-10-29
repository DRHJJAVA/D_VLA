from common14 import *
import numpy as np,h5py,cv2,subprocess,imageio_ffmpeg,traceback
def main():
    assets=ROOT/'docs/图示/cf014_contact';dp=assets/'data.json';data=json.loads(dp.read_text())
    inputs=[dp];configs={}
    for c in data['cases']:
        cp=episode_config(c['id']);cfg=json.loads(cp.read_text());configs[c['id']]=cfg
        inputs.extend([cp,ROOT/cfg['rgbd']])
    save(BASE/'audits/clean_video_inputs_locked.json',lock(inputs))
    records=[]
    for c in data['cases']:
        path=ROOT/configs[c['id']]['rgbd'];vp=assets/(c['id']+'_clean.mp4')
        assert not vp.exists()
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
             '-s','896x448','-r',str(c['fps']),'-i','-','-an','-vcodec','libx264','-crf','18',
             '-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(vp)]
        with open(BASE/(c['id']+'_video.log'),'w') as log,h5py.File(path,'r') as f:
            assert len(f['agentview/rgb'])==c['n'] and len(f['robot0_eye_in_hand/rgb'])==c['n']
            p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=log)
            for t in range(c['n']):
                a=f['agentview/rgb'][t];b=f['robot0_eye_in_hand/rgb'][t]
                if a.shape[:2]!=(448,448):a=cv2.resize(a,(448,448),interpolation=cv2.INTER_LINEAR)
                if b.shape[:2]!=(448,448):b=cv2.resize(b,(448,448),interpolation=cv2.INTER_LINEAR)
                p.stdin.write(np.ascontiguousarray(np.concatenate([a,b],axis=1)).tobytes())
            p.stdin.close();assert p.wait()==0
        cap=cv2.VideoCapture(str(vp));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);ok,img=cap.read();cap.release()
        assert ok and n==c['n'] and abs(fps-c['fps'])<1e-3
        c['src']='图示/cf014_contact/'+vp.name
        records.append(dict(episode=c['id'],frames=n,fps=fps,sha256=sha(vp),source=str(path)))
    # The predictions and all numeric arrays remain identical; only video URLs change.
    old=json.loads(dp.read_text())
    assert all(a['models']==b['models'] and a['metrics']==b['metrics'] for a,b in zip(old['cases'],data['cases']))
    save(assets/'data_before_clean_video.json',old);save(dp,data)
    page=ROOT/'docs/CF014_运动特征与实际接触抓空对照.html'
    s=page.read_text().replace('aspect-ratio:1;max-height:520px','aspect-ratio:2/1;max-height:520px')
    s=s.replace('帧号沿用此前页面，从 0 开始。','左侧为主相机，右侧为腕相机，均为原始 RGB；无旧模型标记。帧号沿用此前页面，从 0 开始。')
    page.write_text(s,encoding='utf-8')
    save(BASE/'audits/clean_video_audit.json',dict(time=now(),records=records,numeric_predictions_unchanged=True,
      reason='existing reusable video contains historical CF005 footer; replace with clean source RGB'))
    save(BASE/'audits/display_manifest.json',dict(time=now(),page=str(page),cases=len(records),
      files=lock([page,dp]+[assets/(c['id']+'_clean.mp4') for c in data['cases']])))
    print(json.dumps(dict(clean_videos=len(records),frames=sum(x['frames'] for x in records))))
if __name__=='__main__':
    try:main()
    except Exception as e:
        save(BASE/'failure_video.json',dict(time=now(),type=type(e).__name__,message=str(e),
          stack=[dict(file=x.filename,line=x.lineno,function=x.name) for x in traceback.extract_tb(e.__traceback__)]))
        raise

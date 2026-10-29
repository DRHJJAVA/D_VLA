from common import *
from features import FrozenSmolFeatures
def main():
 cpu_setup();ds=Dataset();start=time.monotonic()
 for name,h in CFG['p0_source'].items(): assert sha(P/'source'/name)==h
 state('prepare_inputs',complete=0,total=sum(len(v) for v in ds.actions.values()))
 (P/'frames').mkdir(exist_ok=True);(P/'text').mkdir(exist_ok=True)
 enc=FrozenSmolFeatures(str(R/'weights/smolvlm2'),'cuda')
 text,mask=enc.text_features(next(iter(ds.instructions.values())))
 def arr(x):return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
 np.savez(P/'text/train_task.npz',text_features=arr(text),text_valid=arr(mask))
 completed=0;files={};total=sum(len(v) for v in ds.actions.values())
 for eid in CFG['train']+CFG['dev']:
  dst=P/'frames'/eid;dst.mkdir(exist_ok=True)
  for t in range(len(ds.actions[eid])):
   path=dst/(str(t)+'.npz')
   # Full deterministic source frame and frozen features; no trainable outputs cached.
   if not path.exists():
    fr=ds.frame(eid,t)
    sem=arr(enc.point_features(fr['rgb'][None],fr['projected_uv'][None],fr['view_visible'][None]))[0]
    sem[~fr['point_valid']]=0
    assert np.isfinite(sem).all()
    fr=dict(fr);fr['semantic']=sem
    tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**{k:v for k,v in fr.items() if isinstance(v,np.ndarray)});tmp.replace(path)
   files[str(path.relative_to(P))]=sha(path);completed+=1
   if completed%10==0 or completed==total:state('prepare_inputs',complete=completed,total=total,episode=eid,frame=t,seconds=time.monotonic()-start)
 frozen=enc.finish_audit()
 write('inputs.json',{'status':'complete','created':now(),'config_sha256':sha(P/'config.json'),'normalization':ds.normalization,'rows':ds.rows,'frames':files,'text_sha256':sha(P/'text/train_task.npz'),'frozen':frozen,'seconds':time.monotonic()-start,'heldout_read':False})
 state('inputs_complete',complete=completed,total=total,seconds=time.monotonic()-start)
if __name__=='__main__':
 try:main()
 except Exception as e:failure(e);raise SystemExit(1)

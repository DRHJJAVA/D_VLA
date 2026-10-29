import os,sys,pathlib,json,hashlib,datetime,traceback,collections,time,random
import numpy as np,h5py,torch
R=pathlib.Path('/mnt/nvme1/drh/D_VLA/experiments/lcjp003'); P=R/'p1/v001'; ROOT=R.parents[1]
sys.path.insert(0,str(P/'source'))
from data import P0Dataset,INPUT_PROTOCOL_VERSION
CFG=json.loads((P/'config.json').read_text())
KEYS=['semantic','xyz_base','xyz_norm','relation','object_id','point_valid','state','history_valid','group_id']
def now(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def write(name,d):
 p=P/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)
def state(stage,**kw):write('state.json',dict(time=now(),pid=os.getpid(),stage=stage,**kw))
def failure(e):
 write('failure.json',dict(time=now(),pid=os.getpid(),type=type(e).__name__,message=str(e)[:350],frames=[{'file':x.filename,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]))
def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def cpu_setup():
 torch.set_num_threads(CFG['cpu_threads']);torch.set_num_interop_threads(CFG['interop_threads'])
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
class Dataset(P0Dataset):
 def __init__(self,cached=False):
  self.root=ROOT;self.old=ROOT/'experiments/contact_joint_object';self.input_protocol_version=INPUT_PROTOCOL_VERSION
  self.adapter_sha256=sha(P/'source/data.py'); self.read_log=[];self._cache=collections.OrderedDict();self.cached=cached;self.frame_cache=collections.OrderedDict()
  rows=json.loads((self.old/'manifests/episodes.json').read_text())['episodes']
  self.rows={x['id']:x for x in rows if x['id'] in CFG['train']+CFG['dev']}
  assert len(self.rows)==10
  self.actions={};self.instructions={}
  for eid,row in self.rows.items():
   assert row['episode'] in ['demo_'+str(i) for i in list(range(8))+[45,46]]
   with h5py.File(ROOT/row['raw'],'r') as f:
    self.actions[eid]=f['data/'+row['episode']+'/actions'][:].astype(np.float32)
    info=json.loads(f['data'].attrs['problem_info']);self.instructions[eid]=info['language_instruction']
  a=np.concatenate([self.actions[e] for e in CFG['train']])
  self.action_mean=a.mean(0);self.action_std=np.maximum(a.std(0),1e-6)
  self.normalization={'scope':'P1 train0-7 only','mean':self.action_mean.tolist(),'std':self.action_std.tolist(),'count':len(a),'dev_used':False,'clip':False}
 def frame(self,eid,t,n_points=2048):
  assert eid in self.rows
  if not self.cached:return super().frame(eid,t,n_points)
  k=(eid,int(t))
  if k not in self.frame_cache:
   with np.load(P/'frames'/eid/(str(t)+'.npz')) as f:self.frame_cache[k]={x:f[x] for x in f.files}
   if len(self.frame_cache)>512:self.frame_cache.popitem(last=False)
  self.frame_cache.move_to_end(k);return self.frame_cache[k]
 def window(self,eid,t):
  w=self.load_window(eid,t,history=16,horizon=16,n_points=2048)
  real=[self.frame(eid,i)['semantic'] for i in range(max(0,t-15),t+1)]
  sem=np.zeros((16,2048,768),dtype=np.float32);sem[-len(real):]=np.stack(real)
  sem[~w['point_valid']]=0
  w['semantic']=sem
  with np.load(P/'text'/('train_task.npz')) as f:
   w['text_features']=f['text_features'];w['text_valid']=f['text_valid']
  return w
def batch(ds,pairs):
 ws=[ds.window(e,t) for e,t in pairs]
 b={k:torch.from_numpy(np.stack([w[k] for w in ws])).to('cuda',non_blocking=True) for k in KEYS}
 b['text_features']=torch.from_numpy(np.stack([w['text_features'] for w in ws])).to('cuda')
 b['text_valid']=torch.from_numpy(np.stack([w['text_valid'] for w in ws])).to('cuda')
 y=torch.from_numpy(np.stack([w['actions'] for w in ws])).to('cuda')
 m=torch.from_numpy(np.stack([w['action_valid'] for w in ws])).to('cuda')
 return b,y,m

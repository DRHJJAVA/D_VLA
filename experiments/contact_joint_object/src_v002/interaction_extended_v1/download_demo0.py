import os,io,json,time,pathlib,hashlib,requests,h5py,traceback
from datetime import datetime
from zoneinfo import ZoneInfo
BASE=pathlib.Path('/mnt/nvme1/drh/D_VLA')
EXP=BASE/'experiments/contact_joint_object'
RUN=EXP/'runs/v002/preflight/P1_interaction_extended_v1'
RUN.mkdir(parents=True,exist_ok=True)
RAW=RUN/'raw_demo0';RAW.mkdir(exist_ok=True)
FILES=[
('two_objects','libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5','连续抓取字母汤罐和番茄酱罐'),
('drawer_pull','libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5','拉开中层抽屉'),
('drawer_push','libero_90/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5','推回顶层抽屉'),
('stove_switch','libero_goal/turn_on_the_stove_demo.hdf5','操作炉灶开关')]
PROXY={'http':'http://127.0.0.1:19278','https':'http://127.0.0.1:19278'}
REV='147fa5eb607bb696dd922c3518084a90954cd2f3'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def save(p,x):q=p.with_suffix('.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2));q.replace(p)
class RangeFile(io.RawIOBase):
 def __init__(self,url,size):
  self.url=url;self.size=size;self.pos=0;self.cache={};self.block=2**20;self.transferred=0;self.s=requests.Session()
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def read(self,size=-1):
  if size<0:size=self.size-self.pos
  end=min(self.size,self.pos+size);parts=[]
  while self.pos<end:
   k=self.pos//self.block
   if k not in self.cache:
    a=k*self.block;b=min(a+self.block,self.size)-1
    for attempt in range(4):
     try:
      r=self.s.get(self.url+'?download=true&range_start='+str(a),headers={'Range':f'bytes={a}-{b}'},proxies=PROXY,timeout=90)
      assert r.status_code==206 and r.headers.get('Content-Range','').startswith(f'bytes {a}-'), 'HTTP_RANGE_MISMATCH'
      assert len(r.content)==b-a+1,'HTTP_RANGE_LENGTH'
      self.cache[k]=r.content;self.transferred+=len(r.content);break
     except Exception:
      if attempt==3:raise
      time.sleep(2)
   off=self.pos-k*self.block;n=min(end-self.pos,len(self.cache[k])-off)
   parts.append(self.cache[k][off:off+n]);self.pos+=n
  return b''.join(parts)
 def readinto(self,b):
  x=self.read(len(b));b[:len(x)]=x;return len(x)
try:
 meta=requests.get('https://huggingface.co/api/datasets/yifengzhu-hf/LIBERO-datasets/tree/'+REV+'?recursive=true&limit=1000',proxies=PROXY,timeout=40).json()
 lookup={x['path']:x for x in meta}
 protocol={'created':now(),'episodes':[],'scope':'additional user-requested exploratory demo_0 videos; no LCJP-Object heldout reads','source_revision':REV,'association_parameters':'unchanged from P1_interaction_v1','training_updates':0,'task_text_used_for_selection':False,'history':'current and past only'}
 for id,path,label in FILES:
  protocol['episodes'].append({'id':id,'label':label,'path':path,'suite':path.split('/')[0],'episode':'demo_0','source_sha256':lookup[path]['lfs']['oid'],'source_bytes':lookup[path]['size']})
 save(RUN/'protocol.json',protocol)
 md=next((BASE/'docs').glob('LCJP002_*网络训练与前置验证方案.md'))
 with md.open('a') as f:f.write('\n\n### '+now()+' 扩展动作双视角检查\n用户要求增加同视频连续抓取两物体、推/拉抽屉、操作开关。固定选官方 LIBERO 的上述四任务 demo_0；使用完整视频双视角，沿用前轮分割、跟踪与几何运动关联参数，不用任务文本选目标、不训练网络。源文件版本及哈希登记在 P1_interaction_extended_v1/protocol.json，原始示范只读。新结果独立保存，原三段结果不覆盖；开关是否旋转以机构定义和实际画面核对。\n')
 results=[]
 for row in protocol['episodes']:
  start=time.time();out=RAW/(row['id']+'.hdf5')
  if out.exists():continue
  save(RUN/'download_progress.json',{'time':now(),'episode':row['id'],'stage':'reading_demo_0'})
  rf=RangeFile('https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/'+REV+'/'+row['path'],row['source_bytes'])
  with h5py.File(rf,'r') as src,h5py.File(str(out)+'.partial','w') as dst:
   for k,v in src.attrs.items():dst.attrs[k]=v
   d=dst.create_group('data')
   for k,v in src['data'].attrs.items():d.attrs[k]=v
   src.copy(src['data/demo_0'],d,name='demo_0')
   frames=len(d['demo_0/states'])
   schema={k:list(v.shape) for k,v in d['demo_0'].items() if isinstance(v,h5py.Dataset)}
  pathlib.Path(str(out)+'.partial').replace(out)
  os.chmod(out,0o444)
  results.append({'id':row['id'],'frames':frames,'bytes':out.stat().st_size,'range_bytes_transferred':rf.transferred,'seconds':time.time()-start,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'schema':schema})
  save(RUN/'download_results.json',results)
 save(RUN/'download_complete.json',{'time':now(),'episodes':results})
except Exception as e:
 save(RUN/'download_failure.json',{'time':now(),'type':type(e).__name__,'frames':[{'file':pathlib.Path(x.filename).name,'line':x.lineno} for x in traceback.extract_tb(e.__traceback__)]})

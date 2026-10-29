
"""Engineering-only immutable input cache. Original numerical routines remain unchanged."""
import pathlib,json,hashlib,sqlite3,pickle,time,collections,os
import numpy as np
import data
B=data.B
ORIGINAL_GET=data.Dataset.get
ORIGINAL_OBJECTS=data.Dataset.frame_objects
ROOT=B/'cache/engineering_inputs'
def protocol():
    paths=sorted((B/'src').glob('*.py'))+[B/'runs/v001/input_locked.json',B/'runs/v001/measurement_scale_locked.json']
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()
TOKEN=protocol()
ROOT=ROOT/TOKEN
ROOT.mkdir(parents=True,exist_ok=True)
CONNECTIONS=collections.OrderedDict()
STATS={'hit':0,'miss':0,'read_seconds':0.,'compute_seconds':0.,'write_seconds':0.}
def conn(episode):
    if episode in CONNECTIONS:
        CONNECTIONS.move_to_end(episode)
        return CONNECTIONS[episode]
    c=sqlite3.connect(str(ROOT/(episode+'.sqlite')),timeout=120)
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA synchronous=NORMAL')
    c.execute('CREATE TABLE IF NOT EXISTS inputs (key TEXT PRIMARY KEY, value BLOB NOT NULL)')
    CONNECTIONS[episode]=c
    while len(CONNECTIONS)>12:
        _,old=CONNECTIONS.popitem(last=False);old.close()
    return c
def cached_get(self,r,t,oid,ci=0,point_limit=2048):
    key=json.dumps([int(t),int(oid),int(ci),int(point_limit),float(self.sigma)],separators=(',',':'))
    c=conn(r['id']);start=time.perf_counter()
    row=c.execute('SELECT value FROM inputs WHERE key=?',(key,)).fetchone()
    if row is not None:
        raw=pickle.loads(row[0])
        STATS['hit']+=1;STATS['read_seconds']+=time.perf_counter()-start
        return raw
    STATS['miss']+=1
    start=time.perf_counter()
    raw=ORIGINAL_GET(self,r,t,oid,ci=ci,point_limit=point_limit)
    STATS['compute_seconds']+=time.perf_counter()-start
    start=time.perf_counter()
    c.execute('INSERT OR IGNORE INTO inputs VALUES (?,?)',(key,sqlite3.Binary(pickle.dumps(raw,protocol=5))));c.commit()
    STATS['write_seconds']+=time.perf_counter()-start
    return raw
def install():
    data.Dataset.get=cached_get
def close():
    for c in CONNECTIONS.values():c.close()
    CONNECTIONS.clear()
def equivalent(a,b):
    if type(a)!=type(b):return False
    if isinstance(a,np.ndarray):return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b,equal_nan=True)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,float) and np.isnan(a) and np.isnan(b):return True
    return a==b

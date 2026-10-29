import collections,hashlib,torch
class GeometryCache:
 def __init__(self,module,limit_bytes=1024**3):
  self.module=module;self.original={n:getattr(module,n) for n in ['farthest_points','neighbourhood']}
  self.cache=collections.OrderedDict();self.bytes=0;self.limit=limit_bytes;self.hits=0;self.misses=0
 def apply(self):
  for name,fn in self.original.items():
   def wrapper(*args,_name=name,_fn=fn,**kw):
    tensors=[x for x in args if torch.is_tensor(x)]
    if any(x.requires_grad for x in tensors):return _fn(*args,**kw)
    h=hashlib.sha256(_name.encode());device=tensors[0].device
    for x in args:
     if torch.is_tensor(x):
      y=x.detach().contiguous().cpu().numpy();h.update(str((y.shape,y.dtype)).encode());h.update(y.tobytes())
     else:h.update(repr(x).encode())
    h.update(repr(sorted(kw.items())).encode());key=h.digest()
    def move(x,device):return x.to(device).clone() if torch.is_tensor(x) else tuple(move(v,device) for v in x) if isinstance(x,tuple) else [move(v,device) for v in x] if isinstance(x,list) else x
    def size(x):return x.numel()*x.element_size() if torch.is_tensor(x) else sum(size(v) for v in x) if isinstance(x,(list,tuple)) else 0
    if key in self.cache:
     self.hits+=1;self.cache.move_to_end(key);return move(self.cache[key],'cuda')
    self.misses+=1;out=_fn(*args,**kw);stored=move(out,'cpu');self.cache[key]=stored;self.bytes+=size(stored)
    while self.bytes>self.limit:
     _,v=self.cache.popitem(last=False);self.bytes-=size(v)
    return out
   setattr(self.module,name,wrapper)
 def remove(self):
  for k,v in self.original.items():setattr(self.module,k,v)

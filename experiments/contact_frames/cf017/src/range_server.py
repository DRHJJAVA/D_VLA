from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import os,re
ROOT=Path('/mnt/nvme1/drh/D_VLA/docs')
class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*a,**kw):super().__init__(*a,directory=str(ROOT),**kw)
 def send_head(self):
  self.byte_limit=None
  path=Path(self.translate_path(self.path))
  header=self.headers.get('Range')
  if not header or not path.is_file():return super().send_head()
  match=re.fullmatch(r'bytes=(\d*)-(\d*)',header.strip())
  size=path.stat().st_size
  if not match or not size:self.send_error(416);return None
  left,right=match.groups()
  if left:
   start=int(left);end=min(int(right) if right else size-1,size-1)
  elif right:start=max(0,size-int(right));end=size-1
  else:self.send_error(416);return None
  if start>=size or start>end:
   self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return None
  f=path.open('rb');f.seek(start);self.byte_limit=end-start+1
  self.send_response(206);self.send_header('Content-Type',self.guess_type(str(path)));self.send_header('Accept-Ranges','bytes');self.send_header('Content-Range',f'bytes {start}-{end}/{size}');self.send_header('Content-Length',str(self.byte_limit));self.send_header('Last-Modified',self.date_time_string(path.stat().st_mtime));self.end_headers()
  return f
 def end_headers(self):
  if self.byte_limit is None:self.send_header('Accept-Ranges','bytes')
  super().end_headers()
 def copyfile(self,source,output):
  try:
   if self.byte_limit is None:return super().copyfile(source,output)
   remaining=self.byte_limit
   while remaining:
    data=source.read(min(1024*1024,remaining))
    if not data:break
    output.write(data);remaining-=len(data)
  except (BrokenPipeError,ConnectionResetError):pass
if __name__=='__main__':
 ThreadingHTTPServer(('127.0.0.1',19277),Handler).serve_forever()

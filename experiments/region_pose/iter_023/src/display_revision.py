from pathlib import Path
import json,hashlib,html,time
P=Path('/mnt/nvme1/drh/D_VLA');B=P/'experiments/region_pose';O=B/'iter_023';R=B/'reviews/iter_023'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=P/'docs/第023轮_固定旧验证与来源查看.html';new=P/'docs/第023轮_固定旧验证与来源查看_含跨时判定.html'
assert not new.exists()
s=old.read_text();s=s.replace('<nav>','<p class="warn">图标题 endpoint valid 仅指单帧刚体拟合质量。稳定窗口还必须通过下列跨时一致性；白黄杯验证67帧跨时不合格，不能因为单帧拟合通过而使用它生成监督。</p><nav>',1)
manifest=json.loads((R/'manifest.json').read_text())
for m in manifest:
 root=Path(m['source_root']);cases=json.loads((root/m['split']/'summary.json').read_text())['cases'];c=next(c for c in cases if c['id']==m['episode'])
 table='<h3>跨时一致性（不是单帧拟合）</h3><table><tr><th>端点</th><th>判定</th><th>整查询云中位/p95 mm</th><th>实际共同点中位/p95 mm</th><th>原因</th></tr>'
 for t in c.get('time_comparisons',[]):
  fmt=lambda k:format(t[k],'.3f') if isinstance(t.get(k),(int,float)) else '不可用'
  table+='<tr><td>'+html.escape(str(t.get('frames')) )+'</td><td>'+str(t.get('valid'))+'</td><td>'+fmt('median_mm')+' / '+fmt('p95_mm')+'</td><td>'+fmt('observed_median_mm')+' / '+fmt('observed_p95_mm')+'</td><td>'+html.escape(str(t.get('reason')) )+'</td></tr>'
 table+='</table>'
 start=s.index('<section id="'+m['episode']+'">');pos=s.index('<img',start);s=s[:pos]+table+s[pos:]
new.write_text(s,encoding='utf8')
(R/'display_revision.json').write_text(json.dumps({'reason':'显式区分单帧刚体拟合与跨时一致性，避免67帧被误认为稳定窗口。','old_page':str(old),'old_sha256':sha(old),'new_page':str(new),'new_sha256':sha(new),'images_unchanged':True,'new_tracking':False,'time':time.time()},ensure_ascii=False,indent=2))
(R/'display_locked.json').write_text(json.dumps({str(p.relative_to(P)):sha(p) for p in [new,R/'display_revision.json']},indent=2))
print('display_revision_complete')

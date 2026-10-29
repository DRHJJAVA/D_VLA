from pathlib import Path
import json,html,hashlib
import numpy as np
P=Path('/mnt/nvme1/drh/D_VLA');O=P/'experiments/region_pose/iter_017'
def esc(x):return html.escape(str(x))
def table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(f'{v:.4f}' if isinstance(v,float) else v)+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table>'
data={s:json.loads((O/s/'summary.json').read_text()) for s in ['train','val']}
colors=['#0072b2','#0072b2','#d55e00','#d55e00','#009e73','#009e73','#cc79a7','#cc79a7']
def curve(rs,metric):
 out=['<svg viewBox="0 0 680 265" role="img" aria-label="'+esc(metric)+'">']
 for y in [0,.25,.5,.75,1]:
  yy=220-180*y;out.append(f'<path d="M45 {yy} H650" stroke="#ddd"/><text x="4" y="{yy+4}" font-size="12">{y:.2f}</text>')
 for i,b in enumerate([1,2,5,10,20]):
  x=55+i*145;out.append(f'<text x="{x}" y="245" text-anchor="middle" font-size="13">{b}%</text>')
 for j,r in enumerate(rs):
  pts=' '.join(f'{55+i*145},{220-180*c[metric]:.3f}' for i,c in enumerate(r['curves']))
  dash=' stroke-dasharray="5 4"' if j%2 else ''
  out.append(f'<polyline points="{pts}" fill="none" stroke="{colors[j]}" stroke-width="2"{dash}/>')
 return ''.join(out)+'</svg>'
parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>第017轮 冻结模型排序与预算诊断</title><style>body{font:16px system-ui;margin:36px auto;max-width:1300px;color:#192331;background:#f7f8fa}h1,h2,h3{color:#193d52}table{border-collapse:collapse;width:100%;font-size:13px;background:white}td,th{padding:7px;border:1px solid #ddd;text-align:right}td:first-child{text-align:left}details{margin:12px 0}svg{max-width:600px;background:white}section{padding:18px;background:#fff;margin:20px 0}.note{border-left:4px solid #d55e00;padding:12px}</style><h1>第017轮：冻结模型排序与预算诊断</h1><p class="note">真实016冻结best/last模型，共256次查询推理；不是新训练或物理接触精度。训练6事件24查询，旧验证仅2事件8查询。全部输入先排序，未知单列。分数非校准概率。预算1/2/5/10/20%全部预登记展示，不挑验证预算。白黄杯验证及番茄酱验证不足仍未进入指标，第二步未验收。</p><p>蓝late基础、橙late条件、绿multi基础、紫multi条件；实线best、虚线last。multi两检查点均第32轮，曲线重合。横坐标为五个离散预算点，非等间距数值轴。覆盖分母仅已观测正点，未知比例分母为入选点。</p><p><a href="第016轮_真实模型匹配预算对照.html">保留的016固定前20%真实输入与预测图</a></p>']
for split,rr in data.items():
 parts.append('<h2>'+split+'</h2>')
 parts.append(table(['run','ckpt','epoch','AP已知','头loss','训练目标loss'],[[r['run'],r['checkpoint'],r['epoch'],r['overall']['ap'],r['overall']['loss_head'],r['overall']['loss_training_objective']] for r in rr]))
 for ep in ['全部事件等权']+[r['episode'] for r in rr[0]['episodes']]:
  selected=[r['overall'] if ep=='全部事件等权' else next(x for x in r['episodes'] if x['episode']==ep) for r in rr]
  parts.append('<section><h3>'+esc(ep)+'</h3>')
  for metric,label in [('observed_positive_coverage','已观测正点覆盖'),('unknown_fraction','入选未知比例'),('known_negative_fraction','入选已知负例比例')]:
   parts.append('<h4>'+label+'</h4>'+curve(selected,metric))
  parts.append('</section>')
 for r in rr:
  parts.append('<details><summary>'+esc(r['run']+' / '+r['checkpoint'])+'：全部逐查询指标与分数分布</summary>')
  parts.append(table(['query','提前量','AP','头loss','有效率','正score中位','负score中位','未知score中位'],[[x['key'],x['offset'],x['ap'],x['loss_head'],x['valid_fraction']]+[x['distributions'][g]['score']['quantiles']['median'] if x['distributions'][g]['score']['n'] else '无' for g in ['positive','negative','unknown']] for x in r['rows']]))
  for x in r['rows']:
   parts.append('<details><summary>'+esc(x['key'])+'</summary>'+table(['预算%','入选数','正覆盖','入选正','入选负','入选未知','随机期望覆盖'],[[c['budget_percent'],c['selected'],c['observed_positive_coverage'],c['known_positive_fraction'],c['known_negative_fraction'],c['unknown_fraction'],c['random_expected_coverage']] for c in x['curves']])+'<pre>'+esc(json.dumps(x['distributions'],ensure_ascii=False,indent=2))+'</pre></details>')
  parts.append('</details>')
 bas=json.loads((O/split/'baselines.json').read_text())
 parts.append('<details><summary>逐查询随机 / 中心 / 当前夹爪基线（全预算）</summary><pre>'+esc(json.dumps(bas,ensure_ascii=False,indent=2))+'</pre></details>')
parts.append('<h2>历史训练loss（条件含0.5基础辅助loss，不与验证头loss直接相减）</h2>')
history=json.loads((O/'history_analysis.json').read_text())
parts.append(table(['run','1–4','17–20','25–28','29–32','末8轮线性斜率'],[[run]+list(x['blocks'].values())+[x['last8_slope']] for run,x in history.items()]))
parts.append('<p>multi有四轮查询周期。斜率仅描述记录，不能单凭斜率或best轮次认定收敛。全部数字是自动区域监督的一致性；同事件四查询不是四次抓取。</p></html>')
doc=P/'docs/第017轮_冻结模型排序与预算诊断.html';doc.write_text(''.join(parts))
# Independent audit from saved predictions, no GPU or new fitting.
audit=[];freeze=json.loads((O/'pre_validation_freeze.json').read_text())
for split,rr in data.items():
 assert len({x['episode'] for x in rr[0]['rows']})==(6 if split=='train' else 2)
 assert len({x['offset'] for x in rr[0]['rows']})==4
 for r in rr:
  for x in r['rows']:
   path=O/split/'predictions'/r['run']/r['checkpoint']/(x['key']+'.npz')
   with np.load(path) as z:
    y=z['target'];k=z['known'];order=np.argsort(-z['logit'],kind='stable')
    assert not np.any(y&~k) and np.isfinite(z['logit']).all()
    for c in x['curves']:
     n=max(1,int(np.ceil(len(y)*c['budget_percent']/100)));ix=order[:n]
     assert n==c['selected'] and abs(float(y[ix].sum()/y.sum())-c['observed_positive_coverage'])<1e-12
     assert abs(c['known_positive_fraction']+c['known_negative_fraction']+c['unknown_fraction']-1)<1e-12
    if split=='val':assert path.stat().st_mtime>=freeze['time']
   audit.append(str(path.relative_to(P)))
for p,h in freeze['train'].items():assert hashlib.sha256((P/p).read_bytes()).hexdigest()==h
out=dict(prediction_arrays_checked=len(audit),partitions_and_budgets_valid=True,train_frozen_before_val=True,html=str(doc.relative_to(P)),html_sha256=hashlib.sha256(doc.read_bytes()).hexdigest(),old_page_unchanged=True)
(O/'audits/report_audit.json').write_text(json.dumps(out,indent=2))
print(out)

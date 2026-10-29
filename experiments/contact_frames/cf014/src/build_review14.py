from common14 import *
import numpy as np,re,base64,html
from evaluate14 import runs

def main():
    r=ROOT/'docs';assets=r/'图示/cf014_contact';assets.mkdir(parents=True,exist_ok=True)
    oldpage=next(r.glob('CF010*.html'));s=oldpage.read_text()
    match=re.search(r'const\s+clips\s*=\s*(\[)',s)
    if match is None:match=re.search(r'(?:let|var)\s+clips\s*=\s*(\[)',s)
    assert match is not None
    clips,_=json.JSONDecoder().raw_decode(s[match.start(1):])
    metrics=json.loads((BASE/'evaluation/review_metrics.json').read_text())
    cases=[]
    for old in clips:
        ep=old['id'];row=next(x for x in metrics['rows'] if x['episode']==ep)
        src=old['src'];assert src.startswith('data:video/')
        extension='webm' if 'webm' in src[:50] else 'mp4'
        vp=assets/(ep+'.'+extension)
        raw=base64.b64decode(src.split(',',1)[1])
        if vp.exists():assert vp.read_bytes()==raw
        else:vp.write_bytes(raw)
        with np.load(BASE/'evaluation'/ep/'predictions.npz') as d:
            n=d['with_motion'].shape[1];assert n==old['n']
            models={}
            for arm in ['without_motion','with_motion','legacy']:
                a=d[arm];models[arm]=dict(mean=a.mean(0).tolist(),low=a.min(0).tolist(),high=a.max(0).tolist(),ranges=runs(a.mean(0)>=.8))
        cases.append(dict(id=ep,name=old['name'],fps=old['fps'],n=n,
          src='图示/cf014_contact/'+vp.name,human=row['human_positive'],negative=row['human_negative'],
          trained=ep in CFG['train_episodes'],models=models,metrics=row['models']))
    payload=dict(time=now(),cases=cases,aggregate=metrics['aggregate'])
    save(assets/'data.json',payload)
    template=(BASE/'src/review_template.html').read_text()
    page=r/'CF014_运动特征与实际接触抓空对照.html'
    page.write_text(template.replace('__TIME__',now()),encoding='utf-8')
    save(BASE/'audits/display_manifest.json',dict(time=now(),page=str(page),cases=len(cases),
        videos_reused_from=str(oldpage),files=lock([page,assets/'data.json']+[p for p in assets.iterdir() if p.suffix in ['.mp4','.webm']])))
    print(json.dumps(dict(page=str(page),cases=len(cases)),ensure_ascii=False))
if __name__=='__main__':main()

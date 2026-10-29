"""Fixed episode split for Stage 2. Does not inspect or alter demonstration content."""
from pathlib import Path
import json,random,collections
R=Path('/mnt/nvme1/drh/D_VLA')
manifest=R/'data/v3/manifests/original_sequences.jsonl'
rows=[json.loads(x) for x in manifest.read_text().splitlines() if x]
groups=collections.defaultdict(list)
for x in rows:
 groups[x['asset_group']].append(x['id'])
seed=20261023;rng=random.Random(seed)
split={k:[] for k in ('train','val','test')}
for asset,ids in sorted(groups.items()):
 ids=sorted(ids);rng.shuffle(ids)
 if len(ids)<7:raise ValueError('Too few episodes for '+asset)
 split['train']+=ids[:5];split['val']+=ids[5:6];split['test']+=ids[6:]
assert set(sum(split.values(),[]))=={x['id'] for x in rows}
assert len(set(sum(split.values(),[])))==len(rows)
data=dict(seed=seed,unit='whole source episode',asset_groups=sorted(groups),splits=split,counts={k:len(v) for k,v in split.items()},source_manifest=str(manifest),selection_bias='30 episodes previously selected for pick/place; not natural interaction frequency',status='frozen before Stage 2 training')
path=R/'data/v3/manifests/stage2_episode_split.json'
if path.exists():
 assert json.loads(path.read_text())==data,'Existing split differs'
else:path.write_text(json.dumps(data,ensure_ascii=False,indent=2))
print(json.dumps({'counts':data['counts'],'asset_groups':data['asset_groups'],'leakage':False}))

"""Keep two name-free image-change candidates per supplied RGB-D episode.
No geometry truth, actions, object names or labels are read. Original candidates stay intact."""
from pathlib import Path
import json
R=Path('/mnt/nvme1/drh/D_VLA')
rows=[json.loads(x) for x in (R/'data/v3/manifests/original_sequences.jsonl').read_text().splitlines() if x]
split=json.loads((R/'data/v3/manifests/stage2_episode_split.json').read_text())['splits']
lookup={id:k for k,ids in split.items() for id in ids}
assert len(lookup)==len(rows)==30
out=[]
for row in rows:
 path=R/'data/v3/proposals/motion_only'/row['asset_group']/(Path(row['rgbd_path']).stem+'.json')
 q=json.loads(path.read_text())
 assert q['status']=='complete' and q['uses_object_name'] is False and q['uses_simulator_truth'] is False
 out.append(dict(id=row['id'],split=lookup[row['id']],source_rgbd=row['rgbd_path'],frames=row['frames'],candidates=q['candidates'][:2],candidate_count_total=q['candidate_count'],selection='top2 by name-free visual change; both kept without target decision'))
target=R/'data/v3/proposals/motion_only_top2.jsonl'
assert not target.exists(),target
target.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
print(json.dumps(dict(output=str(target),episodes=len(out),counts={k:sum(x['split']==k for x in out) for k in split})))

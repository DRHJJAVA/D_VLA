"""Development-only audit of first-frame template candidates against full-sequence visual ROI.
No simulator truth; query future used only as retrospective reference, never proposal input."""
from pathlib import Path
import json,sys
import numpy as np
R=Path('/mnt/nvme1/drh/D_VLA')
sys.path.insert(0,str(R/'src/v3'))
import stage2_firstframe_template_proposals as template
import stage2_motion_only as motion
import stage2_rerank_rgbd_proposals as rerank
M=R/'data/v3/manifests/stage2_localization_dev_8.json'
OUT=R/'data/v3/proposals/first_frame_templates/dev'
AUDIT=R/'audits/v3/stage2_template_localization_dev_8.json'
def main():
    rows=json.loads(M.read_text())['rows']
    for e in rows:e['frames']=e['source_frames']
    template.E.update({e['id']:e for e in rows})
    donors=template.templates()
    audit=[]
    for e in rows:
        id=e['id'];path=OUT/(id+'.json')
        if path.exists():
            result=json.loads(path.read_text());assert result['no_query_future']
        else:
            result=template.propose(id,donors)
            assert result['candidates']
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(result,indent=2))
        motion.proposals(e)
        reference=rerank.run(e)['selected']['box']
        vals=[template.overlap(c['box'],reference) for c in result['candidates']]
        row=dict(id=id,asset_group=e['asset_group'],reference='full-sequence automatic visual ROI, diagnostic only',candidates=len(vals),top1_iou=vals[0],top5_iou=max(vals[:5]),best_iou=max(vals),best_rank=int(np.argmax(vals))+1,top_scores=[round(x['score'],3) for x in result['candidates'][:5]])
        audit.append(row)
        print(json.dumps(row),flush=True)
    summary=dict(episodes=len(audit),top1_iou_ge_05=sum(x['top1_iou']>=.5 for x in audit),top5_iou_ge_05=sum(x['top5_iou']>=.5 for x in audit),any_iou_ge_05=sum(x['best_iou']>=.5 for x in audit))
    AUDIT.write_text(json.dumps(dict(status='development_diagnostic_only',summary=summary,rows=audit),indent=2))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

from pathlib import Path
import sys,os,ast,hashlib,json,time,subprocess,importlib.util,traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,h5py,torch
from data import B,P,Dataset,to_device
from network import JointModel
from review import build_review
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while True:
   block=f.read(8*1024*1024)
   if not block:break
   h.update(block)
 return h.hexdigest()
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def main():
 freeze=B/'runs/v001/frozen_for_test.json'
 assert freeze.exists(),'No heldout access before training/model freeze'
 lock=json.loads(freeze.read_text());best=B/'runs/v001/best.pt';assert sha(best)==lock['best_sha256']
 for rel,value in lock['source_hashes'].items():assert sha(P/rel)==value,'frozen training source changed'
 records=json.loads((B/'manifests/episodes.json').read_text())['episodes'];row=next(r for r in records if r['id']==lock['test_case'])
 assert row['episode']=='demo_40' and 'bbq_sauce' in row['id'] and row['split']=='heldout'
 out=B/'evaluation/v001';out.mkdir(parents=True,exist_ok=True)
 assert not (out/'test_started.json').exists(),'Test already started; inspect before resuming'
 historical=[str(q.relative_to(P)) for root in [P/'data/rendered',P/'experiments/contact_frames'] for q in root.rglob('*bbq*40*') if q.is_file()]
 (out/'test_started.json').write_text(json.dumps({'time':now(),'case':row['id'],'historical_filename_matches':historical,'blind_test_claim':False,'frozen_before_access':True},indent=2))
 # Test-only access adapter, exact renderer numerical body retained.
 q=B/'src/render_episode.py';tree=ast.parse(q.read_text())
 class Guard(ast.NodeTransformer):
  def visit_Assert(self,n):
   if 'heldout renderer is locked' in ast.unparse(n):
    return ast.parse("assert args.episode=='demo_40' and 'bbq_sauce' in str(args.input) and Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object/runs/v001/frozen_for_test.json').exists()").body[0]
   return n
 adapter=out/'render_test_adapter.py';adapter.write_text(ast.unparse(ast.fix_missing_locations(Guard().visit(tree)))+'\n')
 render=B/'cache/rgbd'/(row['id']+'.h5');env=os.environ.copy();env.update(PYTHONPATH=str(P/'src'),MUJOCO_GL='egl',OMP_NUM_THREADS='2')
 with (B/'logs/test_render.log').open('w') as log:
  proc=subprocess.run([sys.executable,str(adapter),'--input',str(P/row['raw']),'--output',str(render),'--libero-root','/home/drh/project/ECV-VLA/vendor/LIBERO','--episode',row['episode'],'--size','256'],stdout=log,stderr=log,env=env)
 assert proc.returncode==0,'test renderer failed'
 # Only access guard changes; masks/features/tracker numerical implementation is unchanged.
 q=B/'src/observations.py';tree=ast.parse(q.read_text())
 class ObservationGuard(ast.NodeTransformer):
  def visit_Assert(self,n):
   if "row['split']" in ast.unparse(n):
    return ast.parse("assert row['split']=='heldout' and row['id']=='pick_up_the_bbq_sauce_and_place_it_in_the_basket__demo_40' and (B/'runs/v001/frozen_for_test.json').exists()").body[0]
   return n
 adapter2=out/'observation_test_adapter.py';adapter2.write_text(ast.unparse(ast.fix_missing_locations(ObservationGuard().visit(tree)))+'\n')
 spec=importlib.util.spec_from_file_location('lcjp_test_observation',adapter2);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);processor=module.Processor();obs_result=processor.run(row)
 del processor;torch.cuda.empty_cache()
 ds=Dataset('heldout',allow_heldout=True);m=JointModel().cuda().eval();checkpoint=torch.load(best,map_location='cuda',weights_only=False);m.load_state_dict(checkpoint['model'])
 pred=out/'bbq_sauce_demo_40_scores.h5';partial=pred.with_suffix('.partial.h5');values=[];widths=[];input_counts=[];maxdiff=0.
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35)
 allowed=['xyz','normal','features','graph','finger_nodes','finger_normals','finger_side']
 with h5py.File(partial,'w') as h:
  h.attrs.update(best_sha256=sha(best),case=row['id'],current_only=True,output='uncalibrated demonstrated contact preference')
  for t in range(row['frames']):
   frame=h.create_group(str(t));count=0
   for oid in ds.frame_objects(row,t):
    raw=ds.get(row,t,oid);b=to_device({k:raw[k] for k in allowed})
    assert not any(k in b for k in ['context','finger_displacement','motion_target','actions','states'])
    with torch.no_grad():p=m.score_current(b)
    g=frame.create_group(str(oid))
    for name in ['xyz','uv','camera_ids','source_index']:g.create_dataset(name,data=raw[name],compression='gzip')
    for name in ['score','lower','upper']:g.create_dataset(name,data=p[name].cpu().numpy(),compression='gzip')
    scores=p['score'].cpu().numpy();lo=p['lower'].cpu().numpy();hi=p['upper'].cpu().numpy()
    assert np.isfinite(scores).all() and (lo<=scores).all() and (scores<=hi).all()
    values.extend(scores.tolist());widths.extend((hi-lo).tolist());count+=len(scores)
   input_counts.append(count);h.flush()
 partial.replace(pred)
 scores=np.asarray(values);widths=np.asarray(widths)
 summary={'time':now(),'case':row['id'],'split':'heldout after training freeze; not independently certified blind','train_episodes':400,'dev_episodes':50,'epochs_completed':lock['epochs'],'best_epoch':checkpoint['epoch'],'best_sha256':sha(best),'prediction_sha256':sha(pred),'frames':row['frames'],'actual_point_predictions':len(values),'score_quantiles':np.quantile(scores,[0,.1,.5,.9,1]).tolist() if len(scores) else [],'absolute_score_band_counts':np.histogram(scores,[0,.2,.4,.6,.8,1.000001])[0].tolist(),'wide_interval_fraction':float(np.mean(widths>=.8)) if len(widths) else None,'point_counts_per_frame':input_counts,'display_top_fraction':.8,'uses_future_in_current_scoring':False,'input_fields':allowed,'uses_human_contact_ranges':False,'contact_precision_or_success_rate':'not measured; no contact ground truth','score_meaning':'demonstration preference estimate, not current contact probability','limitations':['LIBERO-Object contains pick-and-place only','same-asset heldout episode, not unseen-shape transfer','automatic proposals and correspondences can be uncertain','wide intervals must not be presented as reliable contact']}
 view=build_review(row,pred,render,summary);summary['review']=view;summary['observation_adapter_sha256']=sha(adapter2);summary['renderer_adapter_sha256']=sha(adapter)
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 (B/'review/v001/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 with (P/'docs/无接触标签联合学习_LIBERO-Object网络与训练验证方案.md').open('a') as f:
  f.write('\n\n## E3 冻结测试与实际展示 — '+now()+'\n\n已完成400训练/50开发的'+str(lock['epochs'])+'轮联合运行，按冻结开发观测指标选择第'+str(checkpoint['epoch'])+'轮。冻结后只推理预登记bbq_sauce/demo_40完整视频，共'+str(row['frames'])+'帧、'+str(len(values))+'次实际逐点评分；非人工目标涂色。分数五个绝对区间计数='+str(summary['absolute_score_band_counts'])+'，宽区间占比='+str(summary['wide_interval_fraction'])+'。这些统计不能替代接触精度验收；保留全部低分、未知和失败帧，不以彩色图宣布通过。\n\n[查看实际预测网页]('+Path(view['page']).name+')。默认前80%排名点，固定分数颜色，完整背景和未知可切换，双相机逐帧播放。该案例只支持同资产新视频检查，不能证明新形状或跨动作泛化。\n')
 (B/'experiment_complete.json').write_text(json.dumps({'time':now(),'training_complete':True,'test_prediction_complete':True,'human_acceptance':'pending','page':view['page']},ensure_ascii=False,indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:
  (B/'logs/test_traceback.log').write_text(traceback.format_exc());(B/'test_failure.json').write_text(json.dumps({'time':now(),'exception':type(e).__name__,'detail':str(e)[:300]}));raise

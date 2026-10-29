import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import sys,pathlib,json,time,hashlib,traceback,inspect,datetime
R=pathlib.Path('/mnt/nvme1/drh/D_VLA')
W=R/'experiments/lcjp003/r2/v0_v1/v001'
D=W/'segmentation_diagnostic'
A=R/'docs/LCJP003_R2_V01_assets/coverage32'
sys.path.insert(0,str(R/'experiments/contact_joint_object/dependencies/sam2'))
sys.path.insert(0,str(R/'experiments/contact_joint_object/dependencies/python'))
import numpy as np,torch,h5py
from PIL import Image,ImageDraw,ImageFont
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.utils.amg import MaskData,calculate_stability_score
from torch.nn.attention import sdpa_kernel,SDPBackend

def save(name,obj):
    (D/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cuda.matmul.allow_tf32=False
    lock=json.loads((W/'v1_sampling_lock.json').read_text())
    model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(R/'experiments/contact_joint_object/dependencies/sam21_tiny.pt'),device='cuda')
    reports=[]
    original_init=MaskData.__init__
    original_filter=MaskData.filter
    for ci in [2,6,9]:
        case=lock['cases'][ci]
        assert case['demo'] in ['demo_45','demo_46']
        with h5py.File(R/'experiments/contact_joint_object/cache/rgbd'/(case['id']+'.h5'),'r') as f:
            rgb=f['robot0_eye_in_hand']['rgb'][case['frame']]
            robot=f['robot0_eye_in_hand']['robot_mask'][case['frame']].astype(bool)
        records=[];packed=[];events=[];final_ids=[]
        def traced_init(self,**kwargs):
            original_init(self,**kwargs)
            if 'masks' not in kwargs or 'iou_preds' not in kwargs:return
            masks=kwargs['masks'].detach()
            assert tuple(masks.shape[-2:])==(256,256)
            start=len(records);n=len(masks)
            self['diagnostic_trace_id']=torch.arange(start,start+n,device=masks.device,dtype=torch.int64)
            bits=(masks>0).cpu().numpy()
            quality=kwargs['iou_preds'].detach().cpu().numpy()
            stability=calculate_stability_score(masks,0.,1.).cpu().numpy()
            points=kwargs['points'].detach().cpu().numpy()
            for j in range(n):
                m=bits[j];ys,xs=np.where(m);bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None
                b=np.packbits(m.reshape(-1));packed.append(b)
                records.append({'trace_id':start+j,'point':[float(v) for v in points[j]],'predicted_iou':float(quality[j]),'stability_score':float(stability[j]),'area':int(m.sum()),'bbox':bbox,'first_drop':None,'mask_sha256':hashlib.sha256(b.tobytes()).hexdigest(),'diagnostic_point_128_185':bool(m[185,128]),'diagnostic_roi_pixels':int(m[125:214,45:215].sum())})
        def traced_filter(self,keep):
            ids=self._stats.get('diagnostic_trace_id')
            if ids is not None:
                before=ids.detach().cpu().numpy() if isinstance(ids,torch.Tensor) else np.asarray(ids)
                k=keep.detach().cpu().numpy() if isinstance(keep,torch.Tensor) else np.asarray(keep)
                after=before[k]
                caller=inspect.currentframe().f_back
                method=caller.f_code.co_name;line=caller.f_lineno
                stage={('_process_batch',334):'predicted_iou',('_process_batch',342):'stability',('_process_batch',377):'crop_edge',('_process_crop',285):'box_nms',('_generate_masks',247):'cross_crop_nms'}.get((method,line),method+':'+str(line))
                removed=np.setdiff1d(before,after).astype(int)
                for j in removed:
                    if records[j]['first_drop'] is None:records[j]['first_drop']=stage
                events.append({'stage':stage,'before':len(before),'after':len(after),'removed':len(removed),'caller_line':line})
            original_filter(self,keep)
        MaskData.__init__=traced_init;MaskData.filter=traced_filter
        gen=SAM2AutomaticMaskGenerator(model,points_per_side=32,points_per_batch=64)
        original_generate=gen._generate_masks
        def traced_generate(image):
            data=original_generate(image)
            ids=data._stats.get('diagnostic_trace_id')
            final_ids.extend([int(x) for x in ids])
            return data
        gen._generate_masks=traced_generate
        ts=time.perf_counter()
        try:
            with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
                anns=gen.generate(rgb)
        finally:
            MaskData.__init__=original_init;MaskData.filter=original_filter
        elapsed=time.perf_counter()-ts
        post=[];kept_trace=[]
        assert len(final_ids)==len(anns)
        for tid,ann in zip(final_ids,anns):
            mask=ann['segmentation'].astype(bool)&~robot
            records[tid]['first_drop']='retained' if mask.any() else 'robot_exclusion_empty'
            if mask.any():post.append(mask);kept_trace.append(tid)
        actual=np.stack(post) if post else np.empty((0,256,256),bool)
        expected=np.load(W/('coverage_case'+str(ci)+'_view1_grid32.npz'))['masks']
        same=actual.shape==expected.shape and np.array_equal(actual,expected)
        maxdiff=int(np.count_nonzero(actual!=expected)) if actual.shape==expected.shape else None
        assert same,'instrumented_output_diff_case_'+str(ci)
        np.savez_compressed(D/('case'+str(ci)+'_raw_candidates.npz'),packed=np.stack(packed))
        save('case'+str(ci)+'_candidates.json',{'case':ci,'frame':case['frame'],'target':case['target'],'camera':1,'diagnostic_roi':[45,125,215,214],'roi_is_not_training_label':True,'candidates':records,'filter_events':events,'final_trace_ids':kept_trace})
        counts={}
        for q in records:counts[q['first_drop']]=counts.get(q['first_drop'],0)+1
        containing=[q for q in records if q['diagnostic_point_128_185']]
        unique={}
        for q in sorted(containing,key=lambda x:x['predicted_iou'],reverse=True):
            unique.setdefault(q['mask_sha256'],q)
        chosen=list(unique.values())[:12]
        # Broader masks are shown too; ROI never labels object identity.
        tiles=[]
        for q in chosen:
            m=np.unpackbits(packed[q['trace_id']]).reshape(256,256).astype(bool)
            canvas=rgb.copy();canvas[m]=(canvas[m]*.6+np.array([20,220,170])*.4).astype(np.uint8)
            im=Image.new('RGB',(320,365),'#f4f7fa');im.paste(Image.fromarray(canvas).resize((320,320)),(0,0));draw=ImageDraw.Draw(im)
            draw.text((5,323),'id '+str(q['trace_id'])+'  '+q['first_drop'],fill='#102f40')
            draw.text((5,340),'IoU %.4f / stability %.4f'%(q['predicted_iou'],q['stability_score']),fill='#102f40');tiles.append(im)
        gallery=Image.new('RGB',(4*320,max(1,(len(tiles)+3)//4)*365),'#e1e9ed')
        for j,im in enumerate(tiles):gallery.paste(im,((j%4)*320,(j//4)*365))
        gallery_name='diagnostic_case'+str(ci)+'_prefilter.jpg';gallery.save(A/gallery_name,quality=94)
        report={'case':ci,'frame':case['frame'],'target':case['target'],'camera':1,'generated_candidates':len(records),'final_nonempty':len(post),'stage_counts':counts,'seconds':elapsed,'instrumented_output_exact_match':same,'mask_different_pixels':maxdiff,'point_containing_candidates':len(containing),'top_point_containing':[q for q in chosen],'gallery':'docs/LCJP003_R2_V01_assets/coverage32/'+gallery_name}
        reports.append(report);save('progress.json',{'cases_done':len(reports),'total_cases':3,'latest_case':ci,'optimizer_updates':0})
    save('result.json',{'status':'complete_pending_visual_interpretation','cases':reports,'thresholds_unchanged':True,'points_per_side':32,'pred_iou_thresh':.8,'stability_score_thresh':.95,'box_nms_thresh':.7,'original_source_modified':False,'optimizer_updates':0,'heldout_read':False})
if __name__=='__main__':
    try:main()
    except Exception as e:save('failure.json',{'type':type(e).__name__,'summary':str(e)[:400],'frames':[{'file':f.filename,'line':f.lineno} for f in traceback.extract_tb(e.__traceback__)]})

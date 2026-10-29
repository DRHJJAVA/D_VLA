from pathlib import Path
import sys,os,json,time,traceback
B=Path('/mnt/nvme1/drh/D_VLA/experiments/contact_joint_object');P=B.parents[1]
sys.path[:0]=[str(B/'dependencies/python'),str(B/'dependencies/sam2'),str(P/'src/v3'),str(P/'third_party/co_tracker_010')]
import torch,numpy as np,h5py
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.35)
try:
 from sam2.build_sam import build_sam2
 from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
 from stage2_extract_dino import model,preproc
 from cotracker.predictor import CoTrackerPredictor
 t=time.time()
 sam=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(B/'dependencies/sam21_tiny.pt'),device='cuda')
 gen=SAM2AutomaticMaskGenerator(sam,points_per_side=8,points_per_batch=64)
 dino=model();dino.eval()
 cot=CoTrackerPredictor(checkpoint=str(P/'experiments/region_pose/iter_010/weights/scaled_offline.pth'),offline=True,v2=False,window_len=60).cuda().eval()
 q=B/'cache/rgbd/pick_up_the_alphabet_soup_and_place_it_in_the_basket__demo_0.h5'
 with h5py.File(q,'r') as h:
  im=h['agentview/rgb'][30]
  with torch.inference_mode():
   masks=gen.generate(im)
   feats=dino.forward_features(preproc(im[None]))
   video=torch.from_numpy(h['agentview/rgb'][30:46]).permute(0,3,1,2)[None].float().cuda()
   query=torch.tensor([[[0.,128.,128.],[0.,100.,100.]]],device='cuda')
   tracks,vis=cot(video,queries=query,backward_tracking=False)
 out={'ok':True,'seconds':time.time()-t,'masks':len(masks),'mask_areas':[int(m['segmentation'].sum()) for m in masks],'dino_shape':list(feats.shape),'track_shape':list(tracks.shape),'gpu_peak_bytes':torch.cuda.max_memory_allocated()}
except Exception as e:
 (B/'logs/feature_probe_traceback.log').write_text(traceback.format_exc())
 out={'ok':False,'error':type(e).__name__,'detail':str(e)[:300]}
(B/'manifests/feature_probe.json').write_text(json.dumps(out,indent=2))

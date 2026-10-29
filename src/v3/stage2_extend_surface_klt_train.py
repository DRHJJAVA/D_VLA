"""Extend the frozen KLT/RGB-D pilot to the remaining 17 training episodes only."""
from pathlib import Path
import stage2_track_surface_klt_rgbd as track
import stage2_audit_klt_surface_reprojection as audit
R=Path('/mnt/nvme1/drh/D_VLA')
if __name__=='__main__':
 existing={'tomato_sauce_demo_3','porcelain_mug_demo_4','white_yellow_mug_demo_2'}
 ids=[p.stem for p in sorted((R/'data/v3/proposals/foreground3d_components/train').glob('*.npz')) if p.stem not in existing]
 assert len(ids)==17
 track.SELECTION={'train':ids}
 track.OUT=R/'audits/v3/surface_klt_rgbd_train_extension_v1'
 track.main()
 audit.D=track.OUT
 audit.OUT=R/'audits/v3/surface_klt_rgbd_train_extension_dense_v1'
 audit.main()

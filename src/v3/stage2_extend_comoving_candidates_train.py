from pathlib import Path
import stage2_comoving_occluder_pilot as cue
R=Path('/mnt/nvme1/drh/D_VLA')
if __name__=='__main__':
 cue.BASE=R/'audits/v3/surface_klt_rgbd_train_extension_v1'
 cue.DENSE=R/'audits/v3/surface_klt_rgbd_train_extension_dense_v1'
 cue.OUT=R/'audits/v3/comoving_occluder_train_extension_v1'
 cue.main()

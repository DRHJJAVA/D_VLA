# Experiment status (2026-10-23, Asia/Shanghai)

The active protocol is the original single-frame RGB-D pointwise grasp-region experiment. The previously requested full-video model was trained provisionally, then superseded by the user's correction; its files are archived and its metrics are not used for the active experiment.

Completed:
- Created the `D_VLA` conda environment on the remote data disk.
- Downloaded ten LIBERO-Object task HDF5 files (500 demonstrations, 74,507 states) and checked 10/10 alphabet-soup state restorations.
- Rendered all 50 alphabet-soup demonstrations as synchronized RGB/depth sequences; individual pre-contact frames can be selected without local data transfer.
- Extracted 47 provisional two-finger, lifted-and-stable grasp candidates from 50 alphabet-soup demonstrations. Three episodes did not pass numeric filtering; these must not be treated as negative labels.
- Located cached DINOv2 ViT-S/14 weights on the server and verified they load into the installed timm implementation.

Completed single-frame pilot: generated 20 samples from five training episodes (4,096 points each), with two-camera frozen DINOv2 features, offline object assignment and contact-based soft labels. A pure PyTorch PointNet++ fit ran for 10 epochs on those 20 samples. Balanced training BCE fell from 0.488 to 0.322. On the same training samples, mean AP for points with soft label >0.5 was 0.564 versus 0.00178 positive prevalence; mean AUC was 0.998 (19 of 20 samples had such points). The checkpoint is `outputs/alphabet_soup_pointnet2_fit20/best.pt`; metrics are in `outputs/alphabet_soup_pointnet2_fit20/`. This is a fit check on provisional labels, not a held-out performance result.

Current work: expand single-frame samples across the episode split, review contact labels and image/point alignment, then run held-out training and evaluation.

Quality gates still open: visual RGB/depth/point/segmentation alignment, random review of 30 grasp events, and asset-level split evaluation. The episode split and 20-sample fit check are complete. Provisional training results must not be interpreted as validated grasp performance.

# Experiment status (2026-10-23, Asia/Shanghai)

The active protocol is the original single-frame RGB-D pointwise grasp-region experiment. The previously requested full-video model was trained provisionally, then superseded by the user's correction; its files are archived and its metrics are not used for the active experiment.

Completed:
- Created the `D_VLA` conda environment on the remote data disk.
- Downloaded ten LIBERO-Object task HDF5 files (500 demonstrations, 74,507 states) and checked 10/10 alphabet-soup state restorations.
- Rendered all 50 alphabet-soup demonstrations as synchronized RGB/depth sequences; individual pre-contact frames can be selected without local data transfer.
- Extracted 47 provisional two-finger, lifted-and-stable grasp candidates from 50 alphabet-soup demonstrations. Three episodes did not pass numeric filtering; these must not be treated as negative labels.
- Located cached DINOv2 ViT-S/14 weights on the server and verified they load into the installed timm implementation.
- Selected 30 reproducible candidate events for manual review in `data/audits/contact_review_queue_30.json`; visual review remains pending.

Completed single-frame pilot: generated 20 samples from five training episodes (4,096 points each), with two-camera frozen DINOv2 features, offline object assignment and contact-based soft labels. A pure PyTorch PointNet++ fit ran for 10 epochs on those 20 samples. Balanced training BCE fell from 0.488 to 0.322. On the same training samples, mean AP for points with soft label >0.5 was 0.564 versus 0.00178 positive prevalence; mean AUC was 0.998 (19 of 20 samples had such points). The checkpoint is `outputs/alphabet_soup_pointnet2_fit20/best.pt`; metrics are in `outputs/alphabet_soup_pointnet2_fit20/`. This is a fit check on provisional labels, not a held-out performance result.

Current work: expand single-frame samples across the episode split, review contact labels and image/point alignment, then run held-out training and evaluation.

Quality gates still open: visual RGB/depth/point/segmentation alignment, random review of 30 grasp events, and asset-level split evaluation. The episode split and 20-sample fit check are complete. Provisional training results must not be interpreted as validated grasp performance.

## Episode-held-out 4,096-point result (2026-10-23)

The 33/7/7 episode split yielded 132/28/28 single-frame samples and matching point graphs. An 8-epoch provisional PointNet++ run completed. Validation balanced BCE selected epoch 3 (0.2576); validation mean AP was 0.5619 against a 0.00050 positive-prevalence baseline. Test balanced BCE was 0.2868 and mean AP 0.4672 against 0.00063 prevalence; AP includes only 17/28 test frames with labels >0.5. The checkpoint and exact split are in `outputs/alphabet_soup_pointnet2_heldout/`. No cross-episode split overlap was found. This is performance against automatically generated, unreviewed soft labels, not verified physical grasp success.

Contact-coverage audit at the plan's initial sigma-sized tolerance exposed insufficient object-point density. On validation frames the target has a median 55.5 sampled points (range 23–80); visible contacts have a median 9.17 mm nearest-object-point distance. Selecting every target point covers only 2.98% of visible contact positions after per-episode averaging at sigma tolerance. This is a sampling/evaluation ceiling, so the 4,096-point coverage must not be used to claim model quality. See `docs/coverage_sampling_audit.md`.

A separate 16,384-point technical pilot succeeded on four training frames, including graph construction and forward/backward computation. It raised mean target points from 55.25 to 223 and mean high-label points from 5.5 to 27 on identical frames. `src/run_16k_pipeline.py` is waiting for shared GPU availability before expanding all splits and training a provisional 16k model. Its log is `logs/run_16k_pipeline.log`; it writes only under `data/samples/single_frame_16k` and `outputs/alphabet_soup_pointnet2_16k_provisional`.

Open gates remain unchanged: visual audit of 30 candidate contact events (at least 27 correct), RGB-D/contact-surface alignment, validation-selected contact-coverage tolerance, three-model comparison, and unseen-asset split. The currently reported run does not satisfy these gates.

# Single-frame PointNet++ pilot training

Protocol: `docs/experiment_plan_v1.md`. Training input is one two-camera RGB-D observation converted to 4,096 world-coordinate points and frozen DINOv2 ViT-S/14 patch features (384 dimensions). Model input is `xyz` and `dino_feat` only. Offline `object_ids`, contact events and masks construct the soft labels and supervision mask; they are not fed to the model. The model outputs one logit per point, converted to a score with sigmoid.

Data: 20 pre-contact/start frames from five alphabet-soup demonstrations. Episodes are from the training side of `data/splits/alphabet_soup_episode_split.json`. The training contact library uses only episodes assigned to training. Mean visible target-point count was about 50–80 per 4,096-point sample. One frame had no soft-label point above 0.5; all 20 had at least one above 0.05. These provisional labels still require the planned manual contact review.

Model: frozen official DINOv2 ViT-S/14 weights (SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`), 64-dimension feature compression, PointNet++ set abstraction at 1,024/256/64 points with 4/8/16 cm radii and 32 neighbors, three feature-propagation stages, pointwise 128→64→1 head. Pure PyTorch implementation with cached FPS/radius/3-NN graphs. AdamW, learning rate 0.001, 10 epochs, balanced soft-label BCE. No augmentation.

Fit result: training loss 0.488 → 0.322; mean training AP 0.564 versus random positive prevalence 0.00178; mean training AUC 0.998 over the 19 samples with target >0.5. These are **training-set fit diagnostics**, not validation/test metrics or evidence of unseen-object transfer. Checkpoint: `outputs/alphabet_soup_pointnet2_fit20/best.pt`; epoch logs: `logs/train_pointnet2_fit20.log`; detailed rankings: `outputs/alphabet_soup_pointnet2_fit20/ranking.json`.

Remaining: manually review 30 random contact events and visual alignment, expand samples across train/validation/test with split-isolated label libraries, then run held-out evaluation. The first task has only one target object class and cannot establish unseen-asset transfer.

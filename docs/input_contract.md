# Training input contract

The active experiment follows `docs/experiment_plan_v1.md`. Each model input is one current RGB-D observation from the two cameras, not an entire video. Each frame is converted to a 3D scene point cloud; 4,096 points with frozen DINOv2 pixel features enter the point network. The model outputs one grasp-region score per input point.

Complete demonstrations supply contact-derived supervision and pre-contact frame selection. Instance/element segmentation is used only offline for object assignment and label/audit construction; it is never passed to the model or required at inference. Grasp labels are soft patches around observed successful finger/object contact points, aggregated across training demonstrations of the same asset. The output does not identify paired fingers or label all surfaces between them by default.

The previously run full-video regression trial is archived outside the active source and output directories. Its results are not evidence for this pointwise experiment.

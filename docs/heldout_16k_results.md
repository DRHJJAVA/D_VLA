# Provisional 4k versus 16k known-asset held-out result (2026-10-23)

All samples, weights, metrics, and reports remain on the remote server. Both experiments used the same 33/7/7 episode partition, with 132/28/28 single-frame samples. Each 16k sample has a matching point graph; episode sets are disjoint and identical between the two point-count runs. The 16k pipeline exited normally after eight epochs. Checkpoint and exact metrics are in `outputs/alphabet_soup_pointnet2_16k_provisional/`; machine-readable paired comparison is `comparison_audit.json` there.

| Quantity | 4,096 points | 16,384 points |
|---|---:|---:|
| Median target points, validation | 55.5 | 215.5 |
| Best epoch by validation BCE | 3 | 1 |
| Test mean AP, high soft labels only | 0.4672 (17/28 frames) | 0.2747 (27/28 frames) |
| Test 10 mm sampling ceiling, all target points | 0.3333 | 0.6607 |
| Test coverage at top 10% target points | 0.2440 | 0.4345 |
| Test coverage at top 20% target points | 0.2530 | 0.5060 |
| Test coverage at top 30% target points | 0.2619 | 0.6042 |
| Test random coverage at 20% budget | 0.1079 | 0.3207 |
| Test excess over random at 20% budget | 0.1451 | 0.1852 |
| Test 20%-budget coverage divided by sampling ceiling | 0.759 | 0.766 |

The 10 mm contact-distance tolerance was chosen from validation sampling density and recorded before inspecting the 16k model/test result. Each score uses the same depth-based visibility rule, target-point budget, held-out episodes, and episode-level averaging. On validation, 16k coverage at 20% budget is 0.8452 versus 4k 0.3690; their sampling ceilings are 0.8452 and 0.4583. On test, the paired 20%-budget difference is +0.2530 (descriptive episode-bootstrap 95% interval +0.1071 to +0.4167, seven test episodes). Test visibility fraction averaged by episode is 0.7470 and all 28 frames contain at least one visible candidate contact.

At the original sigma-sized contact tolerance, 16k test coverage at 20% budget is 0.1131 versus random 0.0481; the all-point ceiling is only 0.1399. The 4k values are 0.0595, random 0.0225, and ceiling 0.0595. This narrow tolerance remains dominated by point sampling density.

The absolute 10 mm coverage increase largely reflects denser sampling: the random baseline and all-point ceiling also rise substantially, while model coverage as a fraction of its ceiling stays about 0.76. The lower 16k AP is not directly comparable because the number of frames with >0.5 soft labels changes from 17 to 27 and the label density changes. No claim of improved geometric ranking or grasp success follows from these two pilot runs.

**Remaining protocol gates:** all contact labels are automatic and unreviewed; the selected 30 events require visual checks for correct object, two-finger contact, onset, stable lift, and collision-versus-visual-surface alignment (at least 27 correct). The current loss is a provisional balanced point BCE, not the plan's per-object mean plus 0.1 background term. Checkpoints were selected by validation BCE, not validation contact coverage. Runs use one seed and eight epochs; the plan calls for three seeds, longer training with early stopping, XYZ+RGB PointNet++, XYZ+DINO pointwise MLP, and XYZ+DINO PointNet++ under matched settings. The present dataset contains only one alphabet-soup asset, so no unseen-asset or multi-object conclusion is possible. Prediction artifact export (PLY, raw scores, RGB overlay) is also pending.

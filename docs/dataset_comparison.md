# Dataset comparison

The Phase 2 question is: *how well does one-shot segmentation survive
naturally occurring viewpoint, illumination, scale, background and occlusion
change?* Answering it needs a dataset, so every candidate was checked for the
four properties that decide the answer:

1. **Instance masks** — a target mask is the ground truth; a box-only dataset
   cannot be scored with IoU.
2. **Same-object views** — does the corpus ever show *the same physical object*
   in two frames, or only two different objects of the same class?
3. **Condition labels** — are viewpoint, occlusion, scale or illumination
   annotated, so groups can be compared instead of guessed?
4. **Feasibility here** — annotation bytes and per-file image access, on a
   machine whose sustained link measured **0.17 MB/s** (see *Measurements*).

Every figure below is quoted from the cited source, and the measured column
comes from this machine, not from a marketing page.

## The comparison table

| Dataset | Instance masks | Same-object views | Video | Occlusion | Scale variation | Size | Difficulty |
|---|---|---|---|---|---|---|---|
| **COCO val2017** *(selected)* | yes, 36,781 instances + 446 crowd RLEs in val | no — category-level only | no | unlabeled | natural, unmeasured | 5k val images; annotations 241 MB zip, `instances_val2017.json` 19.1 MB | low–medium: 80 common categories |
| **LVIS v1** | yes, >2M masks | no — category-level only | no | unlabeled | natural, unmeasured | 164k COCO images; long tail, median 9 instances/category | medium–high: 1,208 categories, Zipfian tail |
| **OVIS** | yes, 296k masks | **yes** — 5,223 tracked instances over frames | yes, 901 videos | **labeled** none/slight/severe per frame | natural, annotated every 5 frames | frames 12.7 GB + JSON | high: severe occlusion by design |
| **DAVIS 2017** | yes, per-frame indexed PNG | **yes** — per-object identity across frames | yes, 90 trainval sequences | unlabelled but dense (appearance change, blur) | natural | 480p official; registration required | medium: short clips, 65 max categories |
| **YouTube-VOS 2019** | yes | **yes** — 6,459 tracked instances | yes, 3,471 train videos | unlabelled | natural | 65 categories, 6 fps | medium–high: 65 categories, unseen classes in val |
| **YouTube-VIS 2021** | yes, 232k annotations | **yes** — 8,171 video instances | yes, 3,859 videos | unlabelled | natural | 40 categories | high: full video instance segmentation |
| **PASCAL3D+** | partial — masks only for the VOC2012 subset | no | no | **labeled** occlusion ratio | **labeled** dense viewpoint (azimuth/elevation) | 30,899 instances, 12 rigid categories | medium; masks restricted to the VOC subset |
| **Pascal-Part / PACO** | part-level masks | no | no | unlabeled | natural | multi-GB archives, Drive-hosted | medium; part granularity differs from object |

### What each row means

**COCO** exhaustively annotates all 80 categories across 118k train / 5k val /
41k test images with roughly 1.2M instance masks in total (Gupta et al., 2019,
§5, comparing against COCO). Its val split is the only source here whose
annotations can be fetched as a single 19 MB JSON and whose photographs are
individually downloadable over plain HTTP, which is decisive for the constraints
below.

**LVIS** is COCO imagery annotated for >1,000 entry-level categories with a
deliberately long-tailed, federated annotation scheme — exhaustive *per
constituent dataset*, not per image. It buys vocabulary breadth, not object
identity: two pairs still relate different objects of one category. Its masks
are higher quality than COCO's, which matters when the metric is mask IoU.

**OVIS** is the only corpus here that gives **all three** of same-object
identity, labeled occlusion, and real appearance change: 901 videos, 296k masks,
5,223 instances, annotated every five frames, with frame-level occlusion
annotated as none / slight / severe (converted to 0 / 0.25 / 0.75 and averaged
over the worst half of an instance's frames). Its 12.7 GB frame archive and
2026 move from CodaLab to BaiduPan hosting are what put it out of reach for the
initial build.

**DAVIS 2017** gives per-object identity in 90 trainval sequences at 480p, with
dense per-frame masks as palette-indexed PNGs, and is explicitly designed
around occlusions, motion blur and appearance changes. Access requires
registration, and no ground truth is public for the test-dev / test-challenge
splits.

**YouTube-VOS** and **YouTube-VIS** both carry tracked instance identity
(5,945 unique instances in VOS 2018 train; 8,171 video instances and 232k
annotations in VIS 2021), and VOS marks 26–29 unseen categories for zero-shot
evaluation. They are the natural second stage after same-object matching is
demonstrated to work.

**PASCAL3D+** is the only candidate with *dense, continuous viewpoint*
annotations (azimuth and elevation) plus an occlusion ratio derived from
landmark visibility states. It is also the only one where `viewpoint_condition`
could be filled in honestly for the whole dataset. Its limitation is masks:
3D annotations cover 22,394 ImageNet images as well, but instance masks exist
only for the 8,505 PASCAL images, so a mask-scored benchmark would shrink to
that subset.

## Measurements taken on this machine

The download budget, not the dataset quality, is the binding constraint here.
Measured with ranged HTTP requests against the same COCO archive:

| Transfer | Rate |
|---|---|
| 4 MB range burst | 3.81 MB/s (short, then collapses) |
| 20 MB in ten 2 MB ranges | 0.17 MB/s |
| 19.1 MB single-member fetch | 0.65–0.96 MB/s |
| Full 241 MB archive | abandoned at 22 MB after 15 min |

So a full archive is out, and so is a 12.7 GB frame archive. What *is*
reachable is a single member of a ZIP through `Range` requests — which is how
the builder fetches only `annotations/instances_val2017.json` (6.2 MB
compressed) without ever pulling the other 222 MB — plus per-image JPEGs at
roughly one second each.

## Decision

**Build the initial benchmark from COCO val2017**, with these consequences
stated up front:

- Each pair is **two different instances of the same category in two different
  photographs**, recorded in metadata as `identity_level: "category"`. This is
  category-level transfer, not same-object matching, and results must be read
  as such.
- Viewpoint, illumination and occlusion are recorded as `"unknown"` because
  COCO does not label them. `scale_condition` is the one condition filled from
  measurement (mask area ratio, with thresholds documented in
  `docs/pairing_protocol.md`), because the pixels support it.
- The build is deterministic from `instances_val2017.json` + seed, so the
  100 pairs can be regenerated byte-for-byte.

**Recommended next stage, once a real download window exists: OVIS.** It is the
only source that would let the same protocol be re-run as *genuine* same-object
pairing with a real `occlusion_condition` column, and its categories overlap
COCO's on purpose, so the category-level baseline already built here becomes the
comparison row. DAVIS 2017 is the smaller alternative if registration is
acceptable. PASCAL3D+ is the route to real `viewpoint_condition` labels.

## Citations

- Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context*. ECCV 2014. <https://arxiv.org/abs/1405.0312>
- Gupta, A., Dollar, P., Girshick, R. *LVIS: A Dataset for Large Vocabulary Instance Segmentation*. CVPR 2019. <https://openaccess.thecvf.com/content_CVPR_2019/papers/Gupta_LVIS_A_Dataset_for_Large_Vocabulary_Instance_Segmentation_CVPR_2019_paper.pdf> — dataset site: <https://www.lvisdataset.org/>
- Qi, L. et al. *Occluded Video Instance Segmentation: A Benchmark*. IJCV 2022. <https://link.springer.com/article/10.1007/s11263-022-01629-1> — dataset site: <http://songbai.site/ovis/>
- Perazzi, F. et al. *A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation*. CVPR 2016; Pont-Tuset, J. et al. *The 2017 DAVIS Challenge on Video Object Segmentation*. arXiv:1704.00675. <https://davischallenge.org/challenge2017/rulesdates.html> — downloads: <https://davischallenge.org/davis2017/code.html>
- Xu, N. et al. *YouTube-VOS: A Large-Scale Video Object Segmentation Benchmark*. ECCV 2018. <https://youtube-vos.org/dataset/vos/> — video instance split: <https://youtube-vos.org/dataset/vis/>
- Xiang, Y., Mottaghi, R., Fidler, S. *Beyond PASCAL: A Benchmark for 3D Object Detection in the Wild*. WACV 2014. <https://cvgl.stanford.edu/papers/xiang_wacv14.pdf>
- COCO annotation archives: <http://images.cocodataset.org/annotations/annotations_trainval2017.zip>

## Also considered, not used for masks

Pascal-Part (`http://roozbehm.info/pascal-parts/trainval.tar.gz`) and VOC2010
(`http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar`)
have part-level annotations on multi-gigabyte archives; the Oxford host is
frequently unavailable. The Phase 1 few-shot split files (COCO-20i,
FSS-1000, LVIS-92i, PACO-Part, Pascal-Part) remain hosted on Google Drive and
still need file IDs to fetch, so they could not be used either.

---
library_name: lerobot
base_model: lerobot/pi05_libero_base
pipeline_tag: robotics
tags:
  - lerobot
  - pi05
  - vision-language-action
  - flow-matching
  - libero
  - cloud-edge
license: gemma
---

# CloudEdge π0.5 LIBERO

CloudEdgeVLA-style paired-frame fine-tuning of `lerobot/pi05_libero_base`. The cloud prefix consumes a potentially delayed image; a frozen SigLIP-Base edge encoder consumes the current image and conditions the π0.5 flow-matching action expert.

Training uses 120k optimizer steps, a 21-frame episode-safe history, λmax 0.5 with a 10k-step curriculum, bfloat16, and global batch size 8. These λ/curriculum/batch choices are declared reproduction assumptions because arXiv:2608.00569v1 does not report them.

Evaluation must use the installed `lerobot_policy_cloudedge_pi05` plugin. The implementation source is tracked at [`XiongWenye/EdgeCloud-VLA`](https://github.com/XiongWenye/EdgeCloud-VLA/tree/codex/pi05-cloudedge); the upload job appends the completed result table to this card. This model should not be used for safety-critical or unsupervised physical robot control.

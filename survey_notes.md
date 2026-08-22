# Survey notes

`peng2026cloudedge` demonstrates paired-frame delay training on an OpenVLA-OFT L1-regression system. Its strongest evidence is closed-loop LIBERO success across four suites under a uniform observation-age window. Its real-robot result is a small feasibility pilot on workstation GPUs, not an embedded systems benchmark.

`black2025pi05` supplies the flow-matching VLA architecture and pretrained LIBERO checkpoint used here. The transfer from the paper's L1 action head to a denoising velocity field is an implementation hypothesis tested by this repository; it is not a result claimed by either source.


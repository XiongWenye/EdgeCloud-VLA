# Taxonomy

| Bucket | Evidence | Mechanism | Role here |
|---|---|---|---|
| VLA-specific cloud-edge | `peng2026cloudedge` | Learned stale cloud representation plus current edge vision | Target method |
| VLA flow policy | `black2025pi05` | Conditional flow matching over continuous action chunks | Replacement backbone/action architecture |

This repository tests a direct VLA cloud-edge mechanism, not a generic DNN layer split. The cloud/edge boundary is the π0.5 prefix KV context; the edge retains a frozen current-image encoder and the action expert. No evidence in the two scoped papers establishes Orin/Thor feasibility.


#!/usr/bin/env python3
import argparse

import numpy as np

from openpi.policies import policy_config
from openpi.policies.libero_policy import make_libero_example
from openpi.training import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    policy = policy_config.create_trained_policy(
        config.get_config("pi05_libero"),
        args.checkpoint,
    )
    output = policy.infer(
        make_libero_example(),
        noise=np.zeros((10, 32), dtype=np.float32),
    )
    actions = output["actions"]
    print(
        "OPENPI_MODEL_SMOKE_OK",
        actions.shape,
        bool(np.isfinite(actions).all()),
        output["policy_timing"],
    )


if __name__ == "__main__":
    main()

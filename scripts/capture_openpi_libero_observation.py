#!/usr/bin/env python3
import argparse
import math
import pathlib

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(args.seed)
    try:
        env.reset()
        obs = env.set_init_state(initial_states[args.episode_id])
        for _ in range(10):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        raw_base = np.ascontiguousarray(obs["agentview_image"])
        raw_wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
        base = np.ascontiguousarray(raw_base[::-1, ::-1])
        wrist = np.ascontiguousarray(raw_wrist[::-1, ::-1])
        base = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(base, 224, 224)
        )
        wrist = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, 224, 224)
        )
        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ).astype(np.float32)
        noise = np.random.default_rng(0).standard_normal((10, 32)).astype(np.float32)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            raw_base=raw_base,
            raw_wrist=raw_wrist,
            base=base,
            wrist=wrist,
            state=state,
            prompt=np.asarray(str(task.language)),
            noise=noise,
            suite=np.asarray(args.suite),
            task_id=np.asarray(args.task_id),
            episode_id=np.asarray(args.episode_id),
            seed=np.asarray(args.seed),
            wait_steps=np.asarray(10),
        )
        print(
            "CAPTURE_OK",
            args.output,
            base.shape,
            wrist.shape,
            state.shape,
            task.language,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()

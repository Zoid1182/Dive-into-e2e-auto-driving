import argparse
import csv
import json
from pathlib import Path

import numpy as np


TRAJ_TYPES = ["straight", "left_turn", "right_turn", "accelerate", "decelerate"]


def _make_local_path(rng, traj_type, history_len, future_len, dt):
    total_len = history_len + future_len
    t = (np.arange(total_len) - (history_len - 1)) * dt

    speed0 = rng.uniform(4.0, 12.0)
    yaw0 = rng.uniform(-0.15, 0.15)
    lateral_noise = rng.normal(0.0, 0.03, size=total_len)

    if traj_type == "straight":
        acc = rng.uniform(-0.15, 0.15)
        curvature = rng.uniform(-0.004, 0.004)
    elif traj_type == "left_turn":
        acc = rng.uniform(-0.1, 0.1)
        curvature = rng.uniform(0.018, 0.055)
    elif traj_type == "right_turn":
        acc = rng.uniform(-0.1, 0.1)
        curvature = -rng.uniform(0.018, 0.055)
    elif traj_type == "accelerate":
        acc = rng.uniform(1.0, 2.8)
        curvature = rng.uniform(-0.01, 0.01)
    elif traj_type == "decelerate":
        acc = -rng.uniform(1.0, 2.8)
        curvature = rng.uniform(-0.01, 0.01)
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    ds = np.maximum(speed0 * dt + 0.5 * acc * dt * dt, 0.05)
    step_speed = np.maximum(speed0 + acc * t, 0.2)
    step_dist = step_speed * dt

    x = np.zeros(total_len, dtype=np.float32)
    y = np.zeros(total_len, dtype=np.float32)
    yaw = np.zeros(total_len, dtype=np.float32)
    yaw[0] = yaw0

    for i in range(1, total_len):
        yaw[i] = yaw[i - 1] + curvature * step_dist[i]
        x[i] = x[i - 1] + step_dist[i] * np.cos(yaw[i])
        y[i] = y[i - 1] + step_dist[i] * np.sin(yaw[i])

    y = y + lateral_noise.astype(np.float32)

    anchor = np.array([x[history_len - 1], y[history_len - 1]], dtype=np.float32)
    points = np.stack([x, y], axis=-1).astype(np.float32) - anchor

    # Rotate so the last historical heading is approximately aligned with +x.
    theta = -yaw[history_len - 1]
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    points = points @ rot.T

    params = {
        "speed0_mps": float(speed0),
        "accel_mps2": float(acc),
        "curvature_1pm": float(curvature),
        "yaw0_rad": float(yaw0),
        "dt_s": float(dt),
    }
    return points[:history_len], points[history_len:], params


def make_split(rng, split, n_samples, history_len, future_len, dt):
    histories = np.zeros((n_samples, history_len, 2), dtype=np.float32)
    futures = np.zeros((n_samples, future_len, 2), dtype=np.float32)
    labels = np.zeros((n_samples,), dtype=np.int64)
    rows = []

    for i in range(n_samples):
        label = i % len(TRAJ_TYPES)
        traj_type = TRAJ_TYPES[label]
        history, future, params = _make_local_path(
            rng, traj_type, history_len, future_len, dt
        )
        histories[i] = history
        futures[i] = future
        labels[i] = label
        rows.append(
            {
                "sample_id": f"{split}_{i:06d}",
                "split": split,
                "traj_type": traj_type,
                "label": label,
                **params,
            }
        )

    # Avoid ordered classes on disk while keeping deterministic generation.
    order = rng.permutation(n_samples)
    histories = histories[order]
    futures = futures[order]
    labels = labels[order]
    rows = [rows[j] for j in order]
    for new_i, row in enumerate(rows):
        row["sample_id"] = f"{split}_{new_i:06d}"

    return histories, futures, labels, rows


def save_split(out_dir, split, histories, futures, labels, rows):
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        split_dir / "trajectories.npz",
        history=histories,
        future=futures,
        label=labels,
        label_names=np.array(TRAJ_TYPES),
    )

    with (split_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--future-len", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260617)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train = make_split(
        rng, "train", args.train_samples, args.history_len, args.future_len, args.dt
    )
    test = make_split(
        rng, "test", args.test_samples, args.history_len, args.future_len, args.dt
    )

    save_split(args.out, "train", *train)
    save_split(args.out, "test", *test)

    schema = {
        "name": "synthetic_trajectory_dataset_v1",
        "description": "Toy ego-trajectory prediction dataset for week-1 E2E planning practice.",
        "splits": {
            "train": args.train_samples,
            "test": args.test_samples,
        },
        "dt_s": args.dt,
        "history": {
            "key": "history",
            "shape": [args.history_len, 2],
            "meaning": "Past ego positions in ego-centric frame, meters. Last history point is near [0, 0].",
        },
        "future": {
            "key": "future",
            "shape": [args.future_len, 2],
            "meaning": "Future ego positions in same ego-centric frame, meters.",
        },
        "label": {
            "key": "label",
            "shape": [],
            "mapping": {i: name for i, name in enumerate(TRAJ_TYPES)},
        },
        "files": {
            "train/trajectories.npz": ["history", "future", "label", "label_names"],
            "train/metadata.csv": "sample_id, split, traj_type, label, generation parameters",
            "test/trajectories.npz": ["history", "future", "label", "label_names"],
            "test/metadata.csv": "sample_id, split, traj_type, label, generation parameters",
        },
    }
    with (args.out / "schema.json").open("w") as f:
        json.dump(schema, f, indent=2)

    print(f"Wrote dataset to: {args.out}")
    print(f"Train history shape: {train[0].shape}, future shape: {train[1].shape}")
    print(f"Test history shape: {test[0].shape}, future shape: {test[1].shape}")


if __name__ == "__main__":
    main()

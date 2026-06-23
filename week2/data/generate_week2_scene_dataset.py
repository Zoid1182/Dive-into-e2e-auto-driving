import argparse
import csv
import json
from pathlib import Path

import numpy as np


SCENARIO_TYPES = [
    "straight_clear",
    "left_turn",
    "right_turn",
    "follow_vehicle",
    "obstacle_avoidance",
    "cross_traffic",
]


def _polyline_from_kinematics(rng, history_len, future_len, dt, speed, accel, curvature):
    total_len = history_len + future_len
    yaw = np.zeros(total_len, dtype=np.float32)
    x = np.zeros(total_len, dtype=np.float32)
    y = np.zeros(total_len, dtype=np.float32)

    yaw[0] = rng.uniform(-0.08, 0.08)
    for i in range(1, total_len):
        t = (i - history_len + 1) * dt
        v = max(speed + accel * t, 0.5)
        ds = v * dt
        yaw[i] = yaw[i - 1] + curvature * ds
        x[i] = x[i - 1] + ds * np.cos(yaw[i])
        y[i] = y[i - 1] + ds * np.sin(yaw[i])

    y += rng.normal(0.0, 0.025, size=total_len).astype(np.float32)

    anchor = np.array([x[history_len - 1], y[history_len - 1]], dtype=np.float32)
    points = np.stack([x, y], axis=-1).astype(np.float32) - anchor

    theta = -yaw[history_len - 1]
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    points = points @ rot.T
    return points[:history_len], points[history_len:]


def _make_empty_agents(max_agents):
    agents = np.zeros((max_agents, 5), dtype=np.float32)
    mask = np.zeros((max_agents,), dtype=np.float32)
    return agents, mask


def _add_agent(agents, mask, idx, x, y, vx, vy, radius):
    agents[idx] = np.array([x, y, vx, vy, radius], dtype=np.float32)
    mask[idx] = 1.0


def make_sample(rng, scenario_id, history_len, future_len, max_agents, dt):
    scenario = SCENARIO_TYPES[scenario_id]
    speed = rng.uniform(5.0, 12.0)
    accel = rng.uniform(-0.15, 0.15)
    curvature = rng.uniform(-0.004, 0.004)
    lateral_offset = 0.0
    agents, agent_mask = _make_empty_agents(max_agents)

    if scenario == "straight_clear":
        goal = np.array([rng.uniform(20.0, 32.0), rng.uniform(-0.8, 0.8)], dtype=np.float32)

    elif scenario == "left_turn":
        curvature = rng.uniform(0.018, 0.052)
        goal = np.array([rng.uniform(16.0, 26.0), rng.uniform(5.0, 10.0)], dtype=np.float32)

    elif scenario == "right_turn":
        curvature = -rng.uniform(0.018, 0.052)
        goal = np.array([rng.uniform(16.0, 26.0), -rng.uniform(5.0, 10.0)], dtype=np.float32)

    elif scenario == "follow_vehicle":
        lead_x = rng.uniform(10.0, 22.0)
        lead_y = rng.uniform(-0.4, 0.4)
        lead_speed = max(speed - rng.uniform(1.0, 4.5), 1.0)
        accel = -rng.uniform(0.2, 1.2)
        goal = np.array([rng.uniform(18.0, 32.0), rng.uniform(-0.5, 0.5)], dtype=np.float32)
        _add_agent(agents, agent_mask, 0, lead_x, lead_y, lead_speed, 0.0, rng.uniform(1.0, 1.5))

    elif scenario == "obstacle_avoidance":
        obs_x = rng.uniform(8.0, 18.0)
        obs_y = rng.uniform(-0.35, 0.35)
        side = rng.choice([-1.0, 1.0])
        lateral_offset = side * rng.uniform(1.2, 2.4)
        goal = np.array([rng.uniform(20.0, 32.0), lateral_offset], dtype=np.float32)
        _add_agent(agents, agent_mask, 0, obs_x, obs_y, 0.0, 0.0, rng.uniform(1.0, 1.6))

    elif scenario == "cross_traffic":
        cross_x = rng.uniform(8.0, 18.0)
        start_y = rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 5.5)
        vy = -np.sign(start_y) * rng.uniform(1.5, 4.0)
        accel = -rng.uniform(0.2, 1.0)
        goal = np.array([rng.uniform(18.0, 30.0), rng.uniform(-0.7, 0.7)], dtype=np.float32)
        _add_agent(agents, agent_mask, 0, cross_x, start_y, 0.0, vy, rng.uniform(0.8, 1.3))

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    history, future = _polyline_from_kinematics(
        rng, history_len, future_len, dt, speed, accel, curvature
    )

    if scenario == "obstacle_avoidance":
        # Smooth lateral nudge only in the future horizon.
        phase = np.linspace(0.0, 1.0, future_len, dtype=np.float32)
        future[:, 1] += lateral_offset * np.sin(phase * np.pi / 2.0) ** 1.5

    if scenario == "follow_vehicle":
        # Compress longitudinal progress to mimic following a slower lead vehicle.
        phase = np.linspace(0.0, 1.0, future_len, dtype=np.float32)
        future[:, 0] -= rng.uniform(1.0, 2.8) * phase**2

    if scenario == "cross_traffic":
        # Mild deceleration around the conflict area.
        phase = np.linspace(0.0, 1.0, future_len, dtype=np.float32)
        future[:, 0] -= rng.uniform(0.8, 2.2) * phase**2

    # Add distractor agents away from the ego path.
    n_extra = rng.integers(0, max_agents - 1)
    for j in range(1, 1 + n_extra):
        _add_agent(
            agents,
            agent_mask,
            j,
            rng.uniform(5.0, 30.0),
            rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 8.0),
            rng.uniform(-1.0, 4.0),
            rng.uniform(-0.5, 0.5),
            rng.uniform(0.7, 1.5),
        )

    current_delta = history[-1] - history[-2]
    speed_est = float(np.linalg.norm(current_delta) / dt)
    yaw_est = float(np.arctan2(current_delta[1], current_delta[0]))
    ego_state = np.array([speed_est, yaw_est, accel, curvature], dtype=np.float32)

    return {
        "ego_history": history.astype(np.float32),
        "ego_state": ego_state,
        "goal": goal.astype(np.float32),
        "agents": agents,
        "agent_mask": agent_mask,
        "future": future.astype(np.float32),
        "scenario_type": scenario_id,
        "scenario_name": scenario,
        "speed_mps": speed,
        "accel_mps2": accel,
        "curvature_1pm": curvature,
    }


def make_split(rng, split, n_samples, history_len, future_len, max_agents, dt):
    ego_history = np.zeros((n_samples, history_len, 2), dtype=np.float32)
    ego_state = np.zeros((n_samples, 4), dtype=np.float32)
    goal = np.zeros((n_samples, 2), dtype=np.float32)
    agents = np.zeros((n_samples, max_agents, 5), dtype=np.float32)
    agent_mask = np.zeros((n_samples, max_agents), dtype=np.float32)
    future = np.zeros((n_samples, future_len, 2), dtype=np.float32)
    scenario_type = np.zeros((n_samples,), dtype=np.int64)
    rows = []

    for i in range(n_samples):
        sid = i % len(SCENARIO_TYPES)
        sample = make_sample(rng, sid, history_len, future_len, max_agents, dt)
        ego_history[i] = sample["ego_history"]
        ego_state[i] = sample["ego_state"]
        goal[i] = sample["goal"]
        agents[i] = sample["agents"]
        agent_mask[i] = sample["agent_mask"]
        future[i] = sample["future"]
        scenario_type[i] = sample["scenario_type"]
        rows.append(
            {
                "sample_id": f"{split}_{i:06d}",
                "split": split,
                "scenario_type": sample["scenario_type"],
                "scenario_name": sample["scenario_name"],
                "speed_mps": sample["speed_mps"],
                "accel_mps2": sample["accel_mps2"],
                "curvature_1pm": sample["curvature_1pm"],
                "num_agents": int(sample["agent_mask"].sum()),
                "goal_x_m": float(sample["goal"][0]),
                "goal_y_m": float(sample["goal"][1]),
            }
        )

    order = rng.permutation(n_samples)
    arrays = [
        ego_history[order],
        ego_state[order],
        goal[order],
        agents[order],
        agent_mask[order],
        future[order],
        scenario_type[order],
    ]
    rows = [rows[j] for j in order]
    for i, row in enumerate(rows):
        row["sample_id"] = f"{split}_{i:06d}"
    return arrays, rows


def save_split(out_dir, split, arrays, rows):
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (
        ego_history,
        ego_state,
        goal,
        agents,
        agent_mask,
        future,
        scenario_type,
    ) = arrays

    np.savez_compressed(
        split_dir / "samples.npz",
        ego_history=ego_history,
        ego_state=ego_state,
        goal=goal,
        agents=agents,
        agent_mask=agent_mask,
        future=future,
        scenario_type=scenario_type,
        scenario_names=np.array(SCENARIO_TYPES),
    )

    with (split_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--test-samples", type=int, default=2400)
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--future-len", type=int, default=20)
    parser.add_argument("--max-agents", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train_arrays, train_rows = make_split(
        rng,
        "train",
        args.train_samples,
        args.history_len,
        args.future_len,
        args.max_agents,
        args.dt,
    )
    test_arrays, test_rows = make_split(
        rng,
        "test",
        args.test_samples,
        args.history_len,
        args.future_len,
        args.max_agents,
        args.dt,
    )

    save_split(args.out, "train", train_arrays, train_rows)
    save_split(args.out, "test", test_arrays, test_rows)

    schema = {
        "name": "week2_synthetic_scene_dataset_v1",
        "description": "Scene-aware synthetic trajectory prediction dataset with ego state, route goal, and surrounding agents.",
        "coordinate_frame": "ego-centric; current ego pose is [0, 0], +x points roughly forward, units are meters",
        "dt_s": args.dt,
        "splits": {"train": args.train_samples, "test": args.test_samples},
        "scenario_type_mapping": {i: name for i, name in enumerate(SCENARIO_TYPES)},
        "arrays": {
            "ego_history": {
                "shape": [args.history_len, 2],
                "description": "Past ego positions [x, y]. Last point is [0, 0].",
            },
            "ego_state": {
                "shape": [4],
                "description": "[speed_mps, yaw_rad, accel_mps2, curvature_1pm].",
            },
            "goal": {
                "shape": [2],
                "description": "Route or local target point [x, y].",
            },
            "agents": {
                "shape": [args.max_agents, 5],
                "description": "Padded surrounding agents [x, y, vx, vy, radius].",
            },
            "agent_mask": {
                "shape": [args.max_agents],
                "description": "1 for valid agent rows, 0 for padding rows.",
            },
            "future": {
                "shape": [args.future_len, 2],
                "description": "Future ego trajectory [x, y].",
            },
            "scenario_type": {
                "shape": [],
                "description": "Integer scenario id.",
            },
        },
        "files": {
            "train/samples.npz": "training arrays",
            "train/metadata.csv": "human-readable sample metadata",
            "test/samples.npz": "test arrays",
            "test/metadata.csv": "human-readable sample metadata",
        },
    }
    with (args.out / "schema.json").open("w") as f:
        json.dump(schema, f, indent=2)

    print(f"Wrote dataset to: {args.out}")
    print(f"Train ego_history: {train_arrays[0].shape}, agents: {train_arrays[3].shape}, future: {train_arrays[5].shape}")
    print(f"Test ego_history: {test_arrays[0].shape}, agents: {test_arrays[3].shape}, future: {test_arrays[5].shape}")


if __name__ == "__main__":
    main()

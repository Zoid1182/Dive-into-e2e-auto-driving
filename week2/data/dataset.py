from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np


class MLPDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        data_path = Path(data_dir) / split / "samples.npz"
        data = np.load(data_path)
        # 读取数据
        self.ego_history = data["ego_history"].astype("float32")
        self.ego_state = data["ego_state"].astype("float32")
        self.goal = data["goal"].astype("float32")
        self.agents = data["agents"].astype("float32")
        self.agent_mask = data["agent_mask"].astype("int64")
        self.future = data["future"].astype("float32")
        self.scenario_type = data["scenario_type"].astype("int64")
        self.scenario_names = data["scenario_names"]

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        return {
            "ego_history": torch.from_numpy(self.ego_history[idx]),
            "ego_state": torch.from_numpy(self.ego_state[idx]),
            "goal": torch.from_numpy(self.goal[idx]),
            "agents": torch.from_numpy(self.agents[idx]),
            "agent_mask": torch.from_numpy(self.agent_mask[idx]),
            "future": torch.from_numpy(self.future[idx]),
            "scenario_type": torch.tensor(self.scenario_type[idx])
        }


if __name__ == "__main__":
    dataset = MLPDataset(Path("data/week2_synthetic_scene_dataset_v1"), split="train")
    print(dataset.__getitem__(0))

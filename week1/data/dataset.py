from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np


class MLPDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        data_path = Path(data_dir) / split / "trajectories.npz"
        data = np.load(data_path)
        self.history = data["history"].astype("float32")
        self.future = data["future"].astype("float32")
        self.label = data["label"].astype("int64")
        self.label_names = data["label_names"]

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        return {"history": torch.from_numpy(self.history[idx]),
                "future": torch.from_numpy(self.future[idx]),
                "label": torch.tensor(self.label[idx])}

if __name__ == "__main__":
    dataset = MLPDataset(Path("data/synthetic_trajectory_dataset_v1/train"))
    print(dataset.__getitem__(0))

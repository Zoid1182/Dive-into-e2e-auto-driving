import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajLoss(nn.Module):
    def __init__(self,loss_type="mse"):
        super().__init__()
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

    def forward(self, pred, target):
        return self.loss_fn(pred, target)
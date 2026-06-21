import torch

def ade(pred, target):
  return torch.norm(pred - target, dim=-1).mean()

def fde(pred, target):
  return torch.norm(pred[:, -1] - target[:, -1], dim=-1).mean()
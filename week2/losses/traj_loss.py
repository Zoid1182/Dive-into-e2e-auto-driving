import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajLoss(nn.Module):
    def __init__(
        self,
        future_len=20,
        goal_weight=0.2,
        smooth_weight=0.05,
        collision_weight=0.1,
        ego_radius=0.8,
        safety_margin=0.3,
    ):
        super().__init__()
        
        self.goal_weight = goal_weight
        self.smooth_weight = smooth_weight
        self.collision_weight = collision_weight
        self.ego_radius = ego_radius
        self.safety_margin = safety_margin
        
        weights = torch.linspace(1.0, 2.0, future_len)
        self.register_buffer("time_weights", weights)

    def trajectory_loss(self, pred, target):
        diff = F.smooth_l1_loss(pred, target, reduction="none")
        weights = self.time_weights.view(1, -1, 1)
        return (diff * weights).mean()
    
    def goal_loss(self, pred, batch):
        pred_final = pred[:, -1]
        goal = batch["goal"]
        return F.smooth_l1_loss(pred_final, goal)
    
    def smoothness_loss(self, pred):
        velocity = pred[:, 1:] - pred[:, :-1]
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        return torch.norm(acceleration, dim=-1).mean()
    
    def collision_loss(self, pred, batch):
        agents = batch["agents"]
        agent_mask = batch["agent_mask"]
        agent_pos = agents[..., :2]
        agent_radius = agents[..., 4]
        pred_pos = pred.unsqueeze(2)
        agent_pos = agent_pos.unsqueeze(1)
        dist = torch.norm(pred_pos - agent_pos, dim=-1)
        min_allowed_dist = (
            self.ego_radius
            + agent_radius.unsqueeze(1)
            + self.safety_margin
        )
        violation = F.relu(min_allowed_dist - dist)
        mask = agent_mask.unsqueeze(1)
        violation = violation * mask
        denom = mask.sum() * pred.shape[1] + 1e-6
        return violation.sum() / denom
    
    def forward(self, pred, target, batch):
        traj = self.trajectory_loss(pred, target)
        goal = self.goal_loss(pred, batch)
        smooth = self.smoothness_loss(pred)
        collision = self.collision_loss(pred, batch)
        
        total = (
            traj
            + self.goal_weight * goal
            + self.smooth_weight * smooth
            + self.collision_weight * collision
        )
        return {
            "loss": total,
            "traj_loss": traj.detach(),
            "goal_loss": goal.detach(),
            "smooth_loss": smooth.detach(),
            "collision_loss": collision.detach(),
        }
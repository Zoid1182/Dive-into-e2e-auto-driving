import torch
from torch import nn


class SceneMLP(nn.Module):
    def __init__(
        self,
        history_len=10,
        future_len=20,
        max_agents=8,
        agent_dim=5,
        ego_state_dim=4,
        hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.future_len = future_len
        self.max_agents = max_agents
        self.agent_dim = agent_dim
        
        self.ego_encoder = nn.Sequential(
            nn.Linear(history_len * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.state_encoder = nn.Sequential(
            nn.Linear(ego_state_dim, hidden_dim // 4),
            nn.ReLU()
        )
        
        self.goal_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU()
        )
        
        self.agent_encoder = nn.Sequential(
            nn.Linear(max_agents * agent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        fusion_dim = hidden_dim + hidden_dim // 4 + hidden_dim // 4 + hidden_dim
        
        self.traj_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_len * 2)
        )

    def forward(self, batch):
        ego_history = batch["ego_history"]
        ego_state = batch["ego_state"]
        goal = batch["goal"]
        agents = batch["agents"]
        agent_mask = batch["agent_mask"]
        
        agents = agents * agent_mask.unsqueeze(-1)
        
        batch_size = ego_history.shape[0]
        
        ego_feat = self.ego_encoder(ego_history.reshape(batch_size, -1))
        ego_state_feat = self.state_encoder(ego_state)
        goal_feat = self.goal_encoder(goal)
        agent_feat = self.agent_encoder(agents.reshape(batch_size, -1))
        
        fused_feat = torch.cat(
            [ego_feat, ego_state_feat, goal_feat, agent_feat],
            dim=-1,
        )
        
        pred = self.traj_head(fused_feat)
        return pred.reshape(batch_size, self.future_len, 2)

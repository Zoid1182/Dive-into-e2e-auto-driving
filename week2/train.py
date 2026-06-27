import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data.dataset import MLPDataset
from models.scene_mlp import SceneMLP
from losses.traj_loss import TrajLoss

def move_to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_traj = 0.0
    total_goal = 0.0
    total_smooth = 0.0
    total_collision = 0.0
    total_samples = 0

    for batch in dataloader:
        history = batch["ego_history"].to(device)
        future = batch["future"].to(device)
        batch = move_to_device(batch, device)

        pred = model(batch)
        losses = criterion(pred, future, batch)
        loss = losses["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = history.shape[0]
        total_loss += losses["loss"].item() * batch_size
        total_traj += losses["traj_loss"].item() * batch_size
        total_goal += losses["goal_loss"].item() * batch_size
        total_smooth += losses["smooth_loss"].item() * batch_size
        total_collision += losses["collision_loss"].item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "traj_loss": total_traj / total_samples,
        "goal_loss": total_goal / total_samples,
        "smooth_loss": total_smooth / total_samples,
        "collision_loss": total_collision / total_samples,
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_traj = 0.0
    total_goal = 0.0
    total_smooth = 0.0
    total_collision = 0.0
    total_samples = 0

    for batch in dataloader:
        history = batch["ego_history"].to(device)
        future = batch["future"].to(device)
        batch = move_to_device(batch, device)

        pred = model(batch)
        losses = criterion(pred, future, batch)

        batch_size = history.shape[0]
        total_loss += losses["loss"].item() * batch_size
        total_traj += losses["traj_loss"].item() * batch_size
        total_goal += losses["goal_loss"].item() * batch_size
        total_smooth += losses["smooth_loss"].item() * batch_size
        total_collision += losses["collision_loss"].item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "traj_loss": total_traj / total_samples,
        "goal_loss": total_goal / total_samples,
        "smooth_loss": total_smooth / total_samples,
        "collision_loss": total_collision / total_samples,
    }


def main():
    # input args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/synthetic_trajectory_dataset_v1",
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="outputs/checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    train_dataset = MLPDataset(args.data_root, split="train")
    test_dataset = MLPDataset(args.data_root, split="test")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = SceneMLP(
        history_len=10,
        future_len=20,
        max_agents=8,
        agent_dim=5,
        ego_state_dim=4,
        hidden_dim=128,
        dropout=0.1,
    ).to(device)
    criterion = TrajLoss(
        future_len=20,
        goal_weight=0.2,
        smooth_weight=0.05,
        collision_weight=0.1,
        ego_radius=0.8,
        safety_margin=0.3,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_ade = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"train_traj={train_metrics['traj_loss']:.4f} | "
            f"train_goal={train_metrics['goal_loss']:.4f} | "
            f"train_smooth={train_metrics['smooth_loss']:.4f} | "
            f"train_collision={train_metrics['collision_loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_traj={val_metrics['traj_loss']:.4f} | "
            f"val_goal={val_metrics['goal_loss']:.4f} | "
            f"val_smooth={val_metrics['smooth_loss']:.4f} | "
            f"val_collision={val_metrics['collision_loss']:.4f}"
        )

        if val_metrics["loss"] < best_val_ade:
            best_val_ade = val_metrics["loss"]

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_traj": best_val_ade,
                "args": vars(args),
            }

            torch.save(checkpoint, save_dir / "best_model.pt")
            print(f"Saved best model, val_ADE={best_val_ade:.4f}")

        print("Training finished!")


if __name__ == "__main__":
    main()

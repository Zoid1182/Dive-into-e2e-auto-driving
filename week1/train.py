import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data.dataset import MLPDataset
from models.mlp import TrajMLP
from losses.traj_loss import TrajLoss
from utils.metrics import ade, fde

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0
    
    for batch in dataloader:
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        
        pred = model(history)
        loss = criterion(pred, future)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_size = history.shape[0]
        total_loss += loss.item() * batch_size
        total_ade += ade(pred, future).item() * batch_size
        total_fde += fde(pred, future).item() * batch_size
        total_samples += batch_size
        
    return {
        "loss": total_loss / total_samples,
        "ade": total_ade / total_samples,
        "fde": total_fde / total_samples,
    }

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0
    
    for batch in dataloader:
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        
        pred = model(history)
        loss = criterion(pred, future)
        
        batch_size = history.shape[0]
        total_loss += loss.item() * batch_size
        total_ade += ade(pred, future).item() * batch_size
        total_fde += fde(pred, future).item() * batch_size
        total_samples += batch_size
        
    return {
        "loss": total_loss / total_samples,
        "ade": total_ade / total_samples,
        "fde": total_fde / total_samples,
    }

def main():
    # input args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str,
                        default="data/synthetic_trajectory_dataset_v1", required=True)
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
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = TrajMLP(history_len=10, future_len=20, hidden_dim=128).to(device)
    criterion = TrajLoss(loss_type="smooth_l1")
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
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_ADE={train_metrics['ade']:.4f} "
            f"train_FDE={train_metrics['fde']:.4f} | "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_ADE={val_metrics['ade']:.4f} "
            f"val_FDE={val_metrics['fde']:.4f}"
        )
        
        if val_metrics["ade"] < best_val_ade:
            best_val_ade = val_metrics["ade"]
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_ade": best_val_ade,
                "args": vars(args),
            }
            
            torch.save(checkpoint, save_dir / "best_model.pt")
            print(f"Saved best model, val_ADE={best_val_ade:.4f}")
            
        print("Training finished!")
        
if __name__ == "__main__":
    main()
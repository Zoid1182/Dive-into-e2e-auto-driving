import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from data.dataset import MLPDataset
from models.mlp import TrajMLP
from utils.metrics import ade, fde


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    all_cases = []

    for batch in dataloader:
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        label = batch["label"]

        pred = model(history)

        batch_size = history.shape[0]
        total_ade += ade(pred, future).item() * batch_size
        total_fde += fde(pred, future).item() * batch_size
        total_samples += batch_size
        all_cases.append(
            {
                "history": history.cpu(),
                "future": future.cpu(),
                "pred": pred.cpu(),
                "label": label.cpu(),
            }
        )

        return {
            "ade": total_ade / total_samples,
            "fde": total_fde / total_samples,
            "cases": all_cases,
        }


def plot_case(history, future, pred, save_path, title=""):
    history = history.numpy()
    future = future.numpy()
    pred = pred.numpy()

    plt.plot(
        history[:, 0],
        history[:, 1],
        "o-",
        color="gray",
        label="history",
    )

    plt.plot(
        future[:, 0],
        future[:, 1],
        "o-",
        color="green",
        label="gt future",
    )

    plt.plot(
        pred[:, 0],
        pred[:, 1],
        "o-",
        color="red",
        label="pred future",
    )

    plt.scatter(history[-1, 0], history[-1, 1],
                c="black", s=60, label="current")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.xlabel("x / m")
    plt.ylabel("y / m")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/zoid/projects/Dive-into-e2e-auto-driving/week1/data/synthetic_trajectory_dataset_v1",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/figures",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-vis", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MLPDataset(args.data_root, split="test")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = TrajMLP(
        history_len=10,
        future_len=20,
        hidden_dim=128,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    results = evaluate(model, dataloader, device)
    print(f"Test ADE: {results['ade']:.4f} m")
    print(f"Test FDE: {results['fde']:.4f} m")

    save_dir = Path(args.save_dir)
    label_names = dataset.label_names

    vis_count = 0

    for batch_cases in results["cases"]:
        history_batch = batch_cases["history"]
        future_batch = batch_cases["future"]
        pred_batch = batch_cases["pred"]
        label_batch = batch_cases["label"]

        for i in range(history_batch.shape[0]):
            if vis_count >= args.num_vis:
                break

            history = history_batch[i]
            future = future_batch[i]
            pred = pred_batch[i]
            label = int(label_batch[i])

            case_ade = ade(pred.unsqueeze(0), future.unsqueeze(0)).item()
            case_fde = fde(pred.unsqueeze(0), future.unsqueeze(0)).item()

            title = (
                f"{label_names[label]} | "
                f"ADE={case_ade:.3f}m, FDE={case_fde:.3f}m"
            )

            save_path = save_dir / \
                f"case_{vis_count:03d}_{label_names[label]}.png"
            plot_case(history, future, pred, save_path, title)

            vis_count += 1

        if vis_count >= args.num_vis:
            break

    print(f"Saved {vis_count} figures to {save_dir}")


if __name__ == "__main__":
    main()

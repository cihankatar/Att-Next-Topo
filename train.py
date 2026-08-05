import os
import math
import torch
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.data_loader import loader
from utils.Loss import Dice_CE_Loss
from augmentation.Augmentation import Cutout, cutmix
from wandb_init import parser_init, wandb_init
from utils.metrics import calculate_metrics
from models.Model import ATTNext


def _plot_persistence_diagram(ax, persistence_info, title):
    """Plot H0/H1 diagrams returned by torch_topological for one sample."""
    plotted_values = []
    for dimension, color in ((0, "tab:blue"), (1, "tab:red")):
        if dimension >= len(persistence_info):
            continue
        diagram = persistence_info[dimension][1].detach().cpu()
        if diagram.numel() == 0:
            continue
        diagram = diagram.reshape(-1, 2)
        finite = torch.isfinite(diagram).all(dim=1)
        diagram = diagram[finite]
        if diagram.numel() == 0:
            continue
        values = diagram.numpy()
        ax.scatter(values[:, 0], values[:, 1], s=18, alpha=0.75,
                   color=color, label=f"H{dimension}")
        plotted_values.append(values)

    if plotted_values:
        minimum = min(values.min() for values in plotted_values)
        maximum = max(values.max() for values in plotted_values)
        if minimum == maximum:
            maximum = minimum + 1e-6
        ax.plot([minimum, maximum], [minimum, maximum], "k--", alpha=0.5)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No finite features", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")


def log_topology_example(images, outputs, labels, pi_pred, pi_mask,
                         epoch, image_number, phase, sample_index=0):
    """Log one image, its masks, and corresponding persistence diagrams."""
    image = images[sample_index].detach().cpu()
    if image.shape[0] == 1:
        image = image.squeeze(0).numpy()
        image_cmap = "gray"
    else:
        image = image.permute(1, 2, 0).numpy()
        # Make normalized training images displayable without changing training.
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image_cmap = None

    prediction = torch.sigmoid(outputs[sample_index, 0]).detach().cpu().numpy()
    label = labels[sample_index, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(image, cmap=image_cmap)
    axes[0].set_title("Input")
    axes[1].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Prediction")
    axes[2].imshow(label, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground truth")
    for ax in axes[:3]:
        ax.axis("off")

    _plot_persistence_diagram(axes[3], pi_pred[sample_index], "Prediction PD")
    _plot_persistence_diagram(axes[4], pi_mask[sample_index], "Ground-truth PD")
    fig.suptitle(f"{phase} | epoch {epoch + 1} | image {image_number}")
    fig.tight_layout()

    wandb.log({
        f"{phase}/topology_examples": wandb.Image(fig),
        "epoch": epoch + 1,
        f"{phase}/processed_images": image_number,
    })
    plt.close(fig)

def using_device():
    """Set and print the device used for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    return device

def setup_paths(data):
    """Set up data paths for training and validation."""
    folder_mapping = {
        "isic_2018_1": "isic_1/",
        "kvasir_1": "kvasir_1/",
        "ham_1": "ham_1/",
        "PH2Dataset": "PH2Dataset/",
        "isic_2016_1": "isic_2016_1/"
    }
    folder = folder_mapping.get(data)
    base_path = os.environ["ML_DATA_OUTPUT"] if torch.cuda.is_available() else os.environ["ML_DATA_OUTPUT_LOCAL"]
    return os.path.join(base_path, folder)


# Main Function
def main():
    # Configuration and Initial Setup

    data, training_mode, op, addtopoloss = 'isic_2018_1', "supervised", "train", True
    segmentation_loss_name = "BCE"  # Choose "BCE" or "Dice_BCE".
    topo_lambda = 0.1
    topology_image_size = 32  # 256 -> 128 -> 64 -> 32 (three AvgPool steps)
    visualization_interval = 50

    segmentation_losses = {
        "BCE": "BCE_loss",
        "Dice_BCE": "Dice_BCE_Loss",
    }
    if segmentation_loss_name not in segmentation_losses:
        raise ValueError(
            f"Unknown segmentation loss: {segmentation_loss_name}. "
            f"Choose one of {list(segmentation_losses)}"
        )

    device      = using_device()
    folder_path = setup_paths(data)
    args, res   = parser_init("segmentation task", op, training_mode)
    
    res           = " ".join(res)
    res           = "["+res+"]"
    
    config      = wandb_init(
        os.environ["WANDB_API_KEY"], os.environ["WANDB_DIR"], args, data,
        loss_name=segmentation_loss_name,
        topo_lambda=topo_lambda if addtopoloss else None,
        topology_image_size=topology_image_size if addtopoloss else None,
    )

    # Data Loaders
    def create_loader(operation):
        return loader(operation,args.mode, args.sslmode_modelname, args.bsize, args.workers,args.imsize, args.cutoutpr, args.cutoutbox, args.shuffle, args.sratio, data)

    train_loader    = create_loader(args.op)
    args.op         =  "validation"
    val_loader      = create_loader(args.op)
    args.op         = "train"

    model       = ATTNext(args.mode).to(device)

    topology_name = (
        f"TopoLoss-lam{topo_lambda:g}-{topology_image_size}x{topology_image_size}"
        if addtopoloss else "NoTopoLoss"
    )
    checkpoint_folder_name = (
        f"{model.__class__.__name__}_{segmentation_loss_name}_{topology_name}{res}"
    )
    checkpoint_dir = os.path.join(folder_path, checkpoint_folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save snapshots after 25%, 50%, 75%, and 100% of training. Ceil keeps
    # all four milestones meaningful when the epoch count is not divisible by 4.
    quarter_epochs = [
        math.ceil(config['epochs'] * quarter / 4) for quarter in range(1, 5)
    ]
    optimizer = Adam(model.parameters(), lr=config['learningrate'])
    scheduler = CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['learningrate'] / 10)
    loss_fn   = Dice_CE_Loss()
    segmentation_loss_fn = getattr(loss_fn, segmentation_losses[segmentation_loss_name])

    # checkpoint_path_ssl_read = folder_path+str(model.__class__.__name__)+str(res)
    # model.load_state_dict(torch.load(checkpoint_path_ssl_read, map_location=torch.device('cpu')))

    if addtopoloss:
        from utils.Loss import Topological_Loss
        topo_loss_fn = Topological_Loss(
            lam=topo_lambda,
            topology_image_size=topology_image_size,
        ).to(device)

    print(f"Training on {len(train_loader) * args.bsize} images. Saving checkpoints to {checkpoint_dir}")
    print('Train loader transform',train_loader.dataset.tr)
    print('Val loader transform',val_loader.dataset.tr)
    print(f"Quarter checkpoint epochs: {quarter_epochs}")

    # Training and Validation Loops
    def run_epoch(loader, epoch, training=True):
        """Run a single training or validation epoch."""
        epoch_loss, epoch_loss_, epoch_topo_loss = 0.0, 0.0, 0.0
        model.train() if training else model.eval()

        val_metrics = [0.0] * 5 
        metrics_sum = [0.0] * 5  # To sum up metrics
        num_batches = 0
        processed_images = 0
        next_visualization_at = visualization_interval

        with torch.set_grad_enabled(training):

            for images, labels in tqdm(loader, desc="Training" if training else "Validating", leave=False):
                images, labels = images.to(device), labels.to(device)

                # Apply augmentations during training
                if training and args.aug:
                    images, labels = cutmix(images, labels, args.cutmixpr)
                    images, labels = Cutout(images, labels, args.cutoutpr, args.cutoutbox)
                
                out = model(images)

                loss_ = segmentation_loss_fn(out, labels)

                if addtopoloss:
                    topo_loss,pi_pred,pi_mask = topo_loss_fn(out, labels)
                    total_loss = loss_ + topo_loss
                    epoch_topo_loss += topo_loss.item()
                else:
                    total_loss = loss_

                previous_count = processed_images
                processed_images += images.shape[0]
                if (training and addtopoloss
                        and previous_count < next_visualization_at <= processed_images):
                    log_topology_example(
                        images, out, labels, pi_pred, pi_mask,
                        epoch=epoch,
                        image_number=next_visualization_at,
                        phase="train" if training else "validation",
                        sample_index=next_visualization_at - previous_count - 1,
                    )
                    while next_visualization_at <= processed_images:
                        next_visualization_at += visualization_interval

                epoch_loss += total_loss.item()
                epoch_loss_ += loss_.item()

                # Calculate metrics during validation
                if not training:
                    prediction = (torch.sigmoid(out) > 0.5).float()
                    batch_metrics = calculate_metrics(labels.cpu(), prediction.cpu())
                    metrics_sum = [x + y for x, y in zip(metrics_sum, batch_metrics)]
                    num_batches += 1

                if training:
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()

        if not training and num_batches > 0:
            val_metrics = [x / num_batches for x in metrics_sum]
            return epoch_loss / len(loader), epoch_loss_ / len(loader), epoch_topo_loss / len(loader), val_metrics

        if not training:
            return epoch_loss/len(loader), epoch_loss_/len(loader), epoch_topo_loss/len(loader), val_metrics

        return epoch_loss/len(loader), epoch_loss_/len(loader), epoch_topo_loss/len(loader)

    for epoch in trange(config['epochs'], desc="Epochs"):

        # Training
        train_loss, train_loss_, train_topo_loss = run_epoch(train_loader, epoch, training=True)
        wandb.log({"Train Loss": train_loss, f"Train {segmentation_loss_name} Loss": train_loss_, "Train Topo Loss": train_topo_loss})

        # Validation
        if epoch == 0:
            # Compute validation losses but set metrics to zero
            val_loss, val_loss_, val_topo_loss, _ = run_epoch(val_loader, epoch, training=False)
            val_metrics = [0.0] * 5  # Set metrics to zero
        else:
            val_loss, val_loss_, val_topo_loss, val_metrics = run_epoch(val_loader, epoch, training=False)
            
        wandb.log({
            "Val Loss": val_loss,
            f"Val {segmentation_loss_name} Loss": val_loss_,
            "Val Topo Loss": val_topo_loss,
            "Val IoU": val_metrics[0],
            "Val Dice": val_metrics[1],
            "Val Recall": val_metrics[2],
            "Val Precision": val_metrics[3],
            "Val Accuracy": val_metrics[4],
        })

        # Print losses and validation metrics
        print(f"Epoch {epoch + 1}/{config['epochs']} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train {segmentation_loss_name} Loss: {train_loss_:.4f}, "
              f"Val {segmentation_loss_name} Loss: {val_loss_:.4f}, "
              f"Train Topo Loss: {train_topo_loss:.4f}, Val Topo Loss: {val_topo_loss:.4f}")
        
        print(f"Validation Metrics: IoU: {val_metrics[0]:.4f}, Dice: {val_metrics[1]:.4f}, "
              f"Recall: {val_metrics[2]:.4f}, Precision: {val_metrics[3]:.4f}, "
              f"Accuracy: {val_metrics[4]:.4f}")

        # Save one model snapshot at the end of each training quarter.
        completed_epoch = epoch + 1
        if completed_epoch in quarter_epochs:
            quarter = quarter_epochs.index(completed_epoch) + 1
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"quarter_{quarter}_epoch_{completed_epoch}.pth",
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"Quarter {quarter}/4 checkpoint saved at epoch "
                f"{completed_epoch} with Val Loss: {val_loss:.4f}\n"
                f"Path: {checkpoint_path}"
            )

    wandb.finish()

if __name__ == "__main__":
    main()

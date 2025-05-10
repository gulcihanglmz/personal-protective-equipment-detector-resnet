import os, sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from tqdm import trange, tqdm
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.coco_utils import make_coco_dataset
from models.faster_rcnn import get_model
from engine.train import train_one_epoch
from engine.evaluate import evaluate

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    cfg = yaml.safe_load(open(os.path.join(proj_root, "configs", "default.yaml")))

    # TensorBoard
    writer = SummaryWriter(os.path.join("runs", os.path.basename(__file__).split(".")[0]))

    best_map, patience, no_improve = 0.0, 10, 0
    eval_interval = cfg.get("EVAL_INTERVAL", 5)

    train_ds = make_coco_dataset(cfg["DATA_DIR"], cfg["TRAIN_JSON"], train=True)
    val_ds   = make_coco_dataset(cfg["DATA_DIR"], cfg["VAL_JSON"],   train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )

    # Cihaz ve model
    device = torch.device("cuda" if cfg["DEVICE"]=="cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_model(cfg["NUM_CLASSES"])
    model.to(device)
    model.to(memory_format=torch.channels_last)

    # Windows için compile hatasını atla
    try:
        if hasattr(torch, "compile"):
            model = torch.compile(model)
    except RuntimeError as e:
        tqdm.write(f"[Warning] torch.compile skipped: {e}")

    # Optimizer & Scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["LR"],
        momentum=cfg["MOMENTUM"],
        weight_decay=cfg["WEIGHT_DECAY"]
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["STEP_LR_SIZE"],
        gamma=cfg["STEP_LR_GAMMA"]
    )

    torch.backends.cudnn.benchmark = True

    for epoch in trange(cfg["NUM_EPOCHS"], desc="Epochs"):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        tqdm.write(f"[Epoch {epoch+1}/{cfg['NUM_EPOCHS']}] Train loss: {loss:.4f}")

        lr_scheduler.step()
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if (epoch + 1) % eval_interval == 0:
            stats = evaluate(
                model,
                val_loader,
                os.path.join(cfg["DATA_DIR"], "annotations", cfg["VAL_JSON"]),
                device
            )
            current_map = stats[0]
            writer.add_scalar("mAP/val", current_map, epoch)

            if current_map > best_map:
                best_map, no_improve = current_map, 0
                torch.save(model.state_dict(), "best_model.pth")
                tqdm.write(f" New best mAP: {best_map:.3f} → best_model.pth")
            else:
                no_improve += 1
                tqdm.write(f" {no_improve}/{patience} epochs no improvement")

            if no_improve >= patience:
                tqdm.write(f"Stopping early: no improvement for {patience} epochs.")
                break

        torch.save(model.state_dict(), "last_model.pth")
        torch.cuda.empty_cache()

    writer.close()

if __name__ == "__main__":
    main()

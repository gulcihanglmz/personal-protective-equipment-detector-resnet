import os, sys
proj_root = os.path.abspath(os.path.join(__file__, "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.coco_utils import make_coco_dataset
from models.faster_rcnn import get_model
from engine.train import train_one_epoch
from engine.evaluate import evaluate
from tqdm import trange

def main():

    config_path = os.path.join(proj_root, "configs", "default.yaml")
    cfg = yaml.safe_load(open(config_path))

    # TensorBoard
    log_dir = os.path.join("runs", os.path.basename(__file__).split(".")[0])
    writer  = SummaryWriter(log_dir)

    # Early‐stopping
    best_map   = 0.0
    patience   = 10
    no_improve = 0

    # Dataset + Dataloader
    train_ds = make_coco_dataset(cfg["DATA_DIR"], cfg["TRAIN_JSON"], train=True)
    val_ds   = make_coco_dataset(cfg["DATA_DIR"], cfg["VAL_JSON"],   train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"],
                              shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader   = DataLoader(val_ds,   batch_size=2,
                              shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Model + Opt + LR Scheduler
    device = torch.device("cuda" if cfg["DEVICE"] == "cuda" and torch.cuda.is_available()else "cpu")
    print(f"Using device: {device}")
    model  = get_model(cfg["NUM_CLASSES"])
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["LR"],
                                momentum=cfg["MOMENTUM"],
                                weight_decay=cfg["WEIGHT_DECAY"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=cfg["STEP_LR_SIZE"],
                                                   gamma=cfg["STEP_LR_GAMMA"])


    for epoch in range(cfg["NUM_EPOCHS"]):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"[Epoch {epoch + 1}/{cfg['NUM_EPOCHS']}] Train loss: {loss:.4f}")

        lr_scheduler.step()
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        stats = evaluate(model, val_loader,
                         os.path.join(cfg["DATA_DIR"], "annotations", cfg["VAL_JSON"]),
                         device)

        current_map = stats[0]
        writer.add_scalar("mAP/val", current_map, epoch)

        if current_map > best_map:
            best_map   = current_map
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f" New best mAP: {best_map:.3f}, kaydedildi → best_model.pth")
        else:
            no_improve += 1
            print(f" {no_improve}/{patience} epochs no improvement")

        if no_improve >= patience:
            print(f" No improvement for {patience} epochs—stopping early.")
            break

        torch.save(model.state_dict(), "last_model.pth")
        torch.cuda.empty_cache()

    writer.close()

if __name__ == "__main__":
    main()

import torch
from tqdm import tqdm

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    total_loss = 0

    for images, targets in tqdm(data_loader, desc="Batches", leave=False):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

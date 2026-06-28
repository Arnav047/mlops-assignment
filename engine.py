# engine.py
import torch

def train_one_epoch(model, optimizer, data_loader, device):
    model.train() #  training mode
    total_loss = 0.0

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)


@torch.no_grad() # Turning off gradient calculations to save computer memory
def evaluate_loss(model, data_loader, device):
    model.train() # Faster R-CNN must stay in train mode to calculate loss numbers
    total_loss = 0.0

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

    return total_loss / len(data_loader)
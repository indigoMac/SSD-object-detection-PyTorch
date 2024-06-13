import torch
from torch.utils.data import DataLoader

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        optimizer.step()
        
        total_loss += losses.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}], Batch [{batch_idx}/{len(data_loader)}], Loss: {losses.item():.4f}")
    
    average_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch}] Average Loss: {average_loss:.4f}")

def evaluate(model, data_loader, device):
    model.eval()
    loss_sum = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_sum += losses.item()
    return loss_sum / len(data_loader)

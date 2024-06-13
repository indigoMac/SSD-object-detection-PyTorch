import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader  # Add this line
import config
from dataset import VOCDataset, get_transform
from model import create_ssd_model
from train import train_one_epoch

cudnn.benchmark = True
cudnn.enabled = True

print("Data loading and preprocessing...")
# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define years of VOC datasets to load
years = ['2007', '2008', '2010']

# Load data
train_dataset = VOCDataset('./data', years, 'train', get_transform(train=True))
train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Load validation data without data augmentation
val_dataset = VOCDataset('./data', years, 'val', get_transform(train=False))
val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

print("Done!")

print("Creating SSD model...")
# Create model
model = create_ssd_model(config.num_classes).to(device)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
print("Done!")

print(f"Using device: {device}")

print("Training logic and validation...")
# Training loop
for epoch in range(config.num_epochs):
    train_one_epoch(model, optimizer, train_data_loader, device, epoch)
    print(f"Epoch {epoch+1} completed.")

print("Done!")

print("Saving model...")
torch.save(model.state_dict(), 'ssd_model.pth')
model_save_path = '/content/drive/My Drive/ssd_model/ssd_model.pth'  
torch.save(model.state_dict(), model_save_path)
print("Model saved!")

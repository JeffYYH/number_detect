import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from dataset import HandwritingDataset, train_transform, val_transform
from model import CNN

        
# Load dataset
dataset_dir = 'dataset'
train_dataset = HandwritingDataset(dataset_dir, transform=train_transform)
val_dataset = HandwritingDataset(dataset_dir, transform=val_transform)

# Split dataset
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, optimizer, and scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # Removed verbose

# Training loop with early stopping
num_epochs = 50
best_val_acc = 0.0
patience = 10
counter = 0


epoch_progress = tqdm(range(num_epochs), desc="Training Epochs", position=0)

for epoch in epoch_progress:
    
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
   
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                            position=1, leave=False)
    for inputs, labels in train_progress:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        
        train_acc = correct / total
        train_progress.set_postfix({"loss": f"{train_loss/total:.4f}", "acc": f"{train_acc:.4f}"})
    
    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = correct / total
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    
    val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                        position=1, leave=False)
    with torch.no_grad():
        for inputs, labels in val_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        
            val_acc = correct / total
            val_progress.set_postfix({"loss": f"{val_loss/total:.4f}", "acc": f"{val_acc:.4f}"})
    
    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / total
    
   
    epoch_progress.set_postfix({
        "train_loss": f"{train_loss:.4f}", 
        "train_acc": f"{train_accuracy:.4f}",
        "val_loss": f"{val_loss:.4f}", 
        "val_acc": f"{val_accuracy:.4f}"
    })
    
   
    scheduler.step(val_loss)
    
  
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), 'digit_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("\nEarly stopping triggered")
            break

print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")





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

# 外层进度条：显示总训练轮次
epoch_progress = tqdm(range(num_epochs), desc="Training Epochs", position=0)

for epoch in epoch_progress:
    # 训练阶段
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    # 内层进度条：显示当前轮次的训练批次
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
        
        # 更新进度条信息
        train_acc = correct / total
        train_progress.set_postfix({"loss": f"{train_loss/total:.4f}", "acc": f"{train_acc:.4f}"})
    
    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = correct / total
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    # 内层进度条：显示当前轮次的验证批次
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
            
            # 更新进度条信息
            val_acc = correct / total
            val_progress.set_postfix({"loss": f"{val_loss/total:.4f}", "acc": f"{val_acc:.4f}"})
    
    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / total
    
    # 更新外层进度条信息
    epoch_progress.set_postfix({
        "train_loss": f"{train_loss:.4f}", 
        "train_acc": f"{train_accuracy:.4f}",
        "val_loss": f"{val_loss:.4f}", 
        "val_acc": f"{val_accuracy:.4f}"
    })
    
    # 学习率调度
    scheduler.step(val_loss)
    
    # 早停机制
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




# 使用示例
# model = ...
# train_loader = ...
# val_loader = ...
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
#                    num_epochs=10, patience=3, device=device)    

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * inputs.size(0)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     train_loss = running_loss / len(train_loader.dataset)
#     train_accuracy = correct / total

#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     val_loss = val_loss / len(val_loader.dataset)
#     val_accuracy = correct / total

#     print(f'Epoch {epoch+1}/{num_epochs}, '
#           f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
#           f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

#     # Update scheduler
#     scheduler.step(val_loss)

#     # Early stopping
#     if val_accuracy > best_val_acc:
#         best_val_acc = val_accuracy
#         torch.save(model.state_dict(), 'digit_model.pth')
#         counter = 0
#     else:
#         counter += 1
#         if counter >= patience:
#             print("Early stopping triggered")
#             break

# print(f"Best validation accuracy: {best_val_acc:.4f}")
# print("Model saved to digit_model.pth")
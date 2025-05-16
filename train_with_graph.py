import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming dataset.py defines HandwritingDataset, train_transform, val_transform
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
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Training loop with early stopping
num_epochs = 50
best_val_acc = 0.0
patience = 10
counter = 0

# Lists to store metrics for plotting
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

#show epoch progress
epoch_progress = tqdm(range(num_epochs), desc="Training Epochs", position=0)

for epoch in epoch_progress:
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    #show training progress
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                          position=1, leave=False)
    #training loop
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
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    #show validation progress
    val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                        position=1, leave=False)
    
    #validation loop
    with torch.no_grad():
        for inputs, labels in val_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect predictions and labels for confusion matrix and F1 score
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            val_acc = correct / total
            val_progress.set_postfix({"loss": f"{val_loss/total:.4f}", "acc": f"{val_acc:.4f}"})
    
    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    # Compute F1 score for the epoch
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    epoch_progress.set_postfix({
        "train_loss": f"{train_loss:.4f}", 
        "train_acc": f"{train_accuracy:.4f}",
        "val_loss": f"{val_loss:.4f}", 
        "val_acc": f"{val_accuracy:.4f}",
        "f1_score": f"{f1:.4f}"
    })
    
    #improve learning rate
    scheduler.step(val_loss)
    
    #early stop
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), 'digit_model.pth')
        counter = 0
        # Save best predictions and labels for final confusion matrix
        best_preds = all_preds
        best_labels = all_labels
    else:
        counter += 1
        if counter >= patience:
            print("\nEarly stopping triggered")
            break

print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")

# Plotting
# 1. Training and Validation Loss/Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.close()

# 2. Confusion Matrix
cm = confusion_matrix(best_labels, best_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# 3. Per-class F1 Score
per_class_f1 = f1_score(best_labels, best_preds, average=None)
plt.figure(figsize=(8, 6))
plt.bar(range(10), per_class_f1)
plt.title('Per-Class F1 Score')
plt.xlabel('Class')
plt.ylabel('F1 Score')
plt.xticks(range(10))
plt.grid(True, axis='y')
plt.savefig('f1_scores.png')
plt.close()

# Print final metrics
macro_f1 = f1_score(best_labels, best_preds, average='macro')
print(f"Final Macro F1 Score: {macro_f1:.4f}")
print("Confusion Matrix:")
print(cm)
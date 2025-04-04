import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple
import os

def save_model(model: torch.nn.Module, optimizer: optim.Optimizer, epoch: int, model_path: str = "saved_models/best_model.pth"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)

def calculate_accuracy_SA(model: torch.nn.Module, dataloader: DataLoader, device: str = 'cpu') -> float:
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for texts, labels, lengths in dataloader:
            texts, labels = texts.to(device), labels.to(device).float()
            outputs = model(texts, lengths)  # [batch_size, 1]

            # Convertir logits a probabilidad y luego a predicción binaria
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float().squeeze()  # [batch]
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

def train_torch_model(model: torch.nn.Module, train_dataloader: DataLoader,
                              val_dataloader: DataLoader, criterion: torch.nn.Module,
                              optimizer: optim.Optimizer, epochs: int,
                              print_every: int, patience: int,
                              device: str = 'cpu') -> Tuple[Dict[int, float], Dict[int, float]]:

    writer = SummaryWriter("runs/training_logs")
    train_accuracies, val_accuracies = {}, {}
    best_loss, epochs_no_improve = float('inf'), 0
    model.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        model.train()
        total_loss = 0.0

        for features, labels, text_len in train_dataloader:
            features, labels = features.to(device), labels.to(device).float()  # BCE necesita float
            optimizer.zero_grad()

            outputs = model(features, text_len).squeeze(1)  # [batch_size]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels, text_len in val_dataloader:
                features, labels = features.to(device), labels.to(device).float()
                outputs = model(features, text_len).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # TensorBoard
        writer.add_scalar('Loss/Train', total_loss / len(train_dataloader), epoch)
        writer.add_scalar('Loss/Validation', val_loss / len(val_dataloader), epoch)

        if epoch % print_every == 0 or epoch == epochs - 1:
            train_acc = calculate_accuracy_binary_logits(model, train_dataloader, device)
            val_acc = calculate_accuracy_binary_logits(model, val_dataloader, device)
            train_accuracies[epoch], val_accuracies[epoch] = train_acc, val_acc
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {total_loss / len(train_dataloader):.4f} | "
                  f"Val Loss: {val_loss / len(val_dataloader):.4f} | "
                  f"Train Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}%")

            writer.add_scalar('Accuracy/Train', train_acc * 100, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc * 100, epoch)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            save_model(model, optimizer, epoch)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    writer.close()
    return train_accuracies, val_accuracies

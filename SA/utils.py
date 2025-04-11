import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple
from gensim.models import KeyedVectors

# funciones y clases propias
from LSTM import RNN

# -------- Configuración --------
second_threshold = (0.45, 0.55)

def save_model(model: torch.nn.Module, optimizer: optim.Optimizer, epoch: int, model_path: str = "saved_models/best_model.pth"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)

def load_model(model_path: str, embedding_weights, device: str = "cpu"):
    model = RNN(
        embedding_weights=embedding_weights,
        hidden_dim=128,
        num_layers=3,
        bidirectional=True,
        dropout_p=0.3,
        output_dim=1
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modelo cargado desde {model_path}")
    return model

def load_word2vec(local_path: str ="models/word2vec-google-news-300.kv"):
    """
    Carga el modelo Word2Vec preentrenado desde un archivo local si existe,
    o lo descarga desde Gensim en caso contrario.
    """
    if os.path.exists(local_path):
        print("Cargando modelo Word2Vec desde archivo local...")
        return KeyedVectors.load(local_path)
    else:
        print("Descargando modelo Word2Vec...")
        model = api.load("word2vec-google-news-300")
        # Crear la carpeta "models/" si no existe
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        model.save(local_path)
        return model

def calculate_accuracy_SA(model: torch.nn.Module, dataloader: DataLoader, device: str = 'cpu') -> float:
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for texts, labels, lengths in dataloader:
            texts, labels = texts.to(device), labels.to(device).float()

            outputs = model(texts, lengths)  # [batch_size, 1]
            probs = torch.sigmoid(outputs).squeeze()  # [batch_size]

            # Convertimos las probabilidades en clases 0 o 2
            preds = torch.where(probs >= 0.5, torch.tensor(2.0, device=device), torch.tensor(0.0, device=device))
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

def calculate_accuracy_SA_multiclass(model: torch.nn.Module, dataloader: DataLoader, device: str = 'cpu') -> float:
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for texts, labels, lengths in dataloader:
            texts, labels = texts.to(device), labels.to(device).float()
            outputs = model(texts, lengths)  # [batch_size, 1]

            # Convertir logits a probabilidad
            probs = torch.sigmoid(outputs).squeeze()  # [batch]

            # Asignar clase según el umbral:
            # < 0.4 -> 0 (negativo), 0.4-0.6 -> 1 (neutral), > 0.6 -> 2 (positivo)
            preds = torch.where(
                probs < second_threshold[0], torch.tensor(0, device=device),
                torch.where(probs > second_threshold[1], torch.tensor(2, device=device),
                torch.tensor(1, device=device)))

            # Asegurarse de que las etiquetas también están en el mismo formato (0, 1, 2)
            labels = labels.long()

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
            
            # Normalizamos las etiquetas: 2.0 → 1.0 SOLO para el cálculo del loss
            labels_for_loss = torch.where(labels == 2.0, torch.tensor(1.0, device=device), labels)

            optimizer.zero_grad()
            outputs = model(features, text_len).squeeze(1)
            loss = criterion(outputs, labels_for_loss)  # usamos labels "normalizadas"
            loss.backward()
            optimizer.step()

        # Evaluación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels, text_len in val_dataloader:
                features, labels = features.to(device), labels.to(device).float()
                outputs = model(features, text_len).squeeze(1)
                labels_for_loss = torch.where(labels == 2.0, torch.tensor(1.0, device=device), labels)
                loss = criterion(outputs, labels_for_loss)
                val_loss += loss.item()

        # TensorBoard
        writer.add_scalar('Loss/Train', total_loss / len(train_dataloader), epoch)
        writer.add_scalar('Loss/Validation', val_loss / len(val_dataloader), epoch)

        if epoch % print_every == 0 or epoch == epochs - 1:
            train_acc = calculate_accuracy_SA(model, train_dataloader, device)
            val_acc = calculate_accuracy_SA(model, val_dataloader, device)
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

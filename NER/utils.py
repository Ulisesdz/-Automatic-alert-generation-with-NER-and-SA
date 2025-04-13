import torch
from torch.utils.data import DataLoader
from typing import Dict
import os
from torch.jit import RecursiveScriptModule
import numpy as np
from collections import Counter

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def calculate_confusion_matrix_NER(model: torch.nn.Module, dataloader: DataLoader, tag2idx, device: str = 'cpu') -> None:
    """
    Calculate and display the confusion matrix for NER.
    
    Args:
        model (torch.nn.Module): PyTorch model.
        dataloader (DataLoader): DataLoader with evaluation data.
        tag2idx (dict): Mapping from tag names to indices.
        device (str): 'cpu' or 'cuda'.
    """
    model.to(device)
    model.eval()

    idx2tag = {v: k for k, v in tag2idx.items()}
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sentences, tags, lengths in dataloader:
            sentences, tags = sentences.to(device), tags.to(device)
            logits = model(sentences)  # [batch_size, seq_len, tagset_size]
            predicted_tags = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

            for i in range(len(lengths)):
                pred_seq = predicted_tags[i][:lengths[i]].cpu().tolist()
                label_seq = tags[i][:lengths[i]].cpu().tolist()

                all_preds.extend(pred_seq)
                all_labels.extend(label_seq)

    # Mapeamos índices a etiquetas (opcional, para que la matriz tenga nombres legibles)
    label_names = [idx2tag[i] for i in tag2idx.values() if idx2tag[i] != "<PAD>"]
    label_indices = [i for i in tag2idx.values() if idx2tag[i] != "<PAD>"]

    cm = confusion_matrix(all_labels, all_preds, labels=label_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(xticks_rotation=45, ax=ax, cmap="Blues", colorbar=True)
    plt.title("Matriz de Confusión NER")
    plt.grid(False)
    plt.show()


# Función para calcular los pesos de clase (pesos inversos)
def calculate_class_weights(tag2idx, dataset):
    # Contamos las apariciones de cada etiqueta
    label_counts = Counter()

    for _, tags in dataset:
        label_counts.update(tags.cpu().numpy())
    
    # Total de etiquetas y número de clases
    total_labels = sum(label_counts.values())
    num_classes = len(tag2idx)

    label_counts_dict = dict(label_counts)
    print(label_counts_dict)

    # Devolvemos los pesos inversos en el orden del tag2idx
    weights = []
    for tag in tag2idx.keys():
        print(tag)
        if tag == "<PAD>":
            count = label_counts_dict.get(8, 0)  # Aseguramos que PAD tenga valor 0 si no está presente
        else:
            count = label_counts_dict.get(int(tag), 0)  # Accedemos a las claves como enteros

        # Calculamos los pesos inversos: 1 / frecuencia
        weight = total_labels / (count * num_classes)  # Se añade un pequeño valor para evitar división por 0
        print(f"Etiqueta: {tag}, Frecuencia: {count}, Peso inverso: {weight}")

        weights.append(weight)
    weights[9] = 0.0
    print(weights)
    return torch.tensor(weights, dtype=torch.float)



from collections import defaultdict

def calculate_accuracy_per_tag(model: torch.nn.Module, dataloader: DataLoader, tag2idx, device: str = 'cpu') -> dict:
    """
    Calculate precision, recall, F1 and accuracy per tag for Named Entity Recognition (NER).
    
    Args:
        model (torch.nn.Module): PyTorch model.
        dataloader (DataLoader): DataLoader with batches of data.
        tag2idx (dict): Dictionary mapping tag names to indices.
        device (str): Device ('cpu' or 'cuda').
        
    Returns:
        dict: A dictionary with precision, recall, f1, accuracy, and support per tag.
    """
    model.to(device)
    model.eval()

    idx2tag = {v: k for k, v in tag2idx.items()}

    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    total = defaultdict(int)
    correct = defaultdict(int)

    with torch.no_grad():
        for sentences, tags, lengths in dataloader:
            sentences, tags = sentences.to(device), tags.to(device)
            logits = model(sentences)
            predicted_tags = torch.argmax(logits, dim=-1)

            for i in range(len(lengths)):
                pred_seq = predicted_tags[i][:lengths[i]].cpu().tolist()
                label_seq = tags[i][:lengths[i]].cpu().tolist()

                for pred_tag, true_tag in zip(pred_seq, label_seq):
                    true_tag_name = idx2tag[true_tag]
                    pred_tag_name = idx2tag[pred_tag]

                    if true_tag_name == "<PAD>":
                        continue

                    total[true_tag_name] += 1
                    if pred_tag == true_tag:
                        correct[true_tag_name] += 1
                        true_positives[true_tag_name] += 1
                    else:
                        false_positives[pred_tag_name] += 1
                        false_negatives[true_tag_name] += 1

    metrics = {}
    for tag in total:
        tp = true_positives[tag]
        fp = false_positives[tag]
        fn = false_negatives[tag]
        acc = correct[tag] / total[tag] if total[tag] > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[tag] = {
            "correct": correct[tag],
            "total": total[tag],
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return metrics





def calculate_accuracy_NER(model: torch.nn.Module, dataloader: DataLoader, device: str = 'cpu') -> float:
    """
    Calculate overall token-level accuracy for NER.
    
    Args:
        model (torch.nn.Module): PyTorch model.
        dataloader (DataLoader): DataLoader containing batches of data.
        device (str): Device ('cpu' or 'cuda') where to evaluate the model.
        
    Returns:
        float: The accuracy of the model on the dataset.
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sentences, tags, lengths in dataloader:
            sentences, tags = sentences.to(device), tags.to(device)
            logits = model(sentences)  # [batch_size, seq_len, tagset_size]
            predicted_tags = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

            mask = torch.zeros_like(tags).bool()
            for i in range(len(lengths)):
                mask[i, :lengths[i]] = 1

            correct += ((predicted_tags == tags) & mask).sum().item()
            total += mask.sum().item()
    
    return correct / total if total > 0 else 0.0


def train_torch_model(model: torch.nn.Module, train_dataloader: DataLoader,
                      val_dataloader: DataLoader, criterion: torch.nn.Module,
                      optimizer: torch.optim.Optimizer, epochs: int,
                      print_every: int, patience: int,
                      device: str = 'cpu') -> Dict[int, float]:
    """
    Train and validate the NER model with early stopping.

    Args:
        model (torch.nn.Module): PyTorch model to train.
        train_dataloader (DataLoader): Training dataset loader.
        val_dataloader (DataLoader): Validation dataset loader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Number of epochs.
        print_every (int): Frequency of printing training stats.
        patience (int): Early stopping patience.
        device (str): Device to train on ('cpu' or 'cuda').

    Returns:
        Dict[int, float]: Dictionary with training and validation accuracies.
    """
    train_accuracies = {}
    val_accuracies = {}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for sentences, tags, lengths in train_dataloader:
            sentences, tags = sentences.to(device), tags.to(device)
            optimizer.zero_grad()
            outputs = model(sentences)  # [batch_size, seq_len, tagset_size]

            # Flatten for loss calculation: ignore padding with view
            loss = criterion(outputs.view(-1, outputs.shape[-1]), tags.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sentences, tags, lengths in val_dataloader:
                sentences, tags = sentences.to(device), tags.to(device)
                outputs = model(sentences)
                loss = criterion(outputs.view(-1, outputs.shape[-1]), tags.view(-1))
                val_loss += loss.item()

        # Logging
        if epoch % print_every == 0 or epoch == epochs - 1:
            train_acc = calculate_accuracy_NER(model, train_dataloader, device)
            val_acc = calculate_accuracy_NER(model, val_dataloader, device)
            train_accuracies[epoch] = train_acc
            val_accuracies[epoch] = val_acc
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss / len(train_dataloader):.4f} | "
                  f"Val Loss: {val_loss / len(val_dataloader):.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_model(model, "bilstm_model.pth")  # Changed file name to reflect model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break
        print(epoch)
        
    return train_accuracies, val_accuracies



def save_model(model: torch.nn.Module, name: str) -> None:
    if not os.path.isdir("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), f"models/{name}")  # NO .pt extra

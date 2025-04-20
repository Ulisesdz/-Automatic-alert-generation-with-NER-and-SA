import torch
from torch.utils.data import DataLoader
import os
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from collections import defaultdict

from LSTM import BiLSTM
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
            logits = model(sentences, lengths)  # [batch_size, seq_len, tagset_size]
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
def calculate_class_weights_sklearn(tag2idx, dataset):
    all_tags = []

    for tags in dataset.ner_tags:
        all_tags.extend(tags)

    # Eliminar las etiquetas de padding (-1)
    all_tags = [tag for tag in all_tags if tag != -1]

    classes = list(tag2idx.values())
    classes = [c for c in classes if c != -1]  # quitamos el índice del padding

    # Calcular los pesos
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=all_tags)

    # Crear un tensor de pesos donde el índice corresponde a cada clase
    weight_tensor = torch.zeros(len(tag2idx), dtype=torch.float)

    for i, cls in enumerate(classes):
        weight_tensor[cls] = class_weights[i]

    # Asegúrate de que el índice de padding tenga peso 0
    if -1 in tag2idx.values():
        pad_idx = [k for k, v in tag2idx.items() if v == -1][0]
        weight_tensor[tag2idx[pad_idx]] = 0.0

    return weight_tensor

def calculate_accuracy_per_tag(model: torch.nn.Module, dataloader: DataLoader, tag2idx, device: str = 'cpu') -> dict:
    """
    Calculate accuracy per tag for Named Entity Recognition (NER).
    
    Args:
        model (torch.nn.Module): PyTorch model.
        dataloader (DataLoader): DataLoader with batches of data.
        tag2idx (dict): Dictionary mapping tag names to indices.
        device (str): Device ('cpu' or 'cuda').
        
    Returns:
        dict: A dictionary with accuracy per tag.
    """
    model.to(device)
    model.eval()

    idx2tag = {v: k for k, v in tag2idx.items()}

    total = defaultdict(int)
    correct = defaultdict(int)

    with torch.no_grad():
        for sentences, tags, lengths in dataloader:
            sentences, tags = sentences.to(device), tags.to(device)
            logits = model(sentences, lengths)
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

    accuracy_per_tag = {}
    for tag in total:
        acc = correct[tag] / total[tag] if total[tag] > 0 else 0.0
        accuracy_per_tag[tag] = {
            "accuracy": acc,
            "correct": correct[tag],
            "total": total[tag]
        }

    return accuracy_per_tag


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
            logits = model(sentences, lengths)  # [batch_size, seq_len, tagset_size]
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
                      device: str = 'cpu') -> dict[int, float]:
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
        print(f"Epoch: {epoch+1}")
        model.train()
        total_loss = 0.0
        for sentences, tags, lengths in train_dataloader:
            sentences, tags = sentences.to(device), tags.to(device)
            optimizer.zero_grad()
            
            outputs = model(sentences, lengths)  # [B, T, C]

            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tags.reshape(-1))  # Flatten both
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sentences, tags, lengths in val_dataloader:
                sentences, tags = sentences.to(device), tags.to(device)
                outputs = model(sentences, lengths)
                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tags.reshape(-1))
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
            save_model(model, f"mismo_1.pth")  
        else:
            epochs_no_improve += 1
            patience_left = patience - epochs_no_improve
            print(f"Validation loss did not improve. Patience left: {patience_left}")
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break
        
    return train_accuracies, val_accuracies


def save_model(model, model_path: str):
    # Guardar el modelo completo
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(BASE_DIR, "saved_models", model_path)
    torch.save(model, full_path)

def load_model(model_path: str, device: str = 'cpu'):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(BASE_DIR, "saved_models", model_path)
    model = torch.load(full_path)
    model.to(device)
    model.eval()  # Establecer el modelo en modo evaluación
    return model

def evaluate(rnn_model, train_dataloader, val_dataloader, test_dataloader, device, full_train_dataset):
    # Final evaluation on train, validation, and test datasets
    train_acc = calculate_accuracy_NER(rnn_model, train_dataloader, device=device)
    val_acc = calculate_accuracy_NER(rnn_model, val_dataloader, device=device)
    test_acc = calculate_accuracy_NER(rnn_model, test_dataloader, device=device)

    # Print results
    print(f"\n🔹 NER Model - Training Accuracy: {train_acc:.4f}")
    print(f"🔹 NER Model - Validation Accuracy: {val_acc:.4f}")
    print(f"🔹 NER Model - Test Accuracy: {test_acc:.4f}")

    tag_accuracy = calculate_accuracy_per_tag(rnn_model, test_dataloader, full_train_dataset.tag2idx, device)

    print("\n🔍 Accuracy por etiqueta (NER):")
    for tag in sorted(tag_accuracy.keys(), key=lambda t: full_train_dataset.tag2idx[t]):
        info = tag_accuracy[tag]
        print(f"{str(full_train_dataset.tag2idx[tag]):8s} → {info['accuracy']:.4f}  (Correctas: {info['correct']}, Totales: {info['total']})")


    # Después de entrenar el modelo y realizar la evaluación
    calculate_confusion_matrix_NER(rnn_model, test_dataloader, full_train_dataset.tag2idx, device=device)


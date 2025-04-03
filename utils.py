import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Dict, Tuple


def calculate_accuracy_SA(model: torch.nn.Module, dataloader: DataLoader, device: str = 'cpu') -> float:
    """
    Calcula la precisión (accuracy) de un modelo PyTorch dado un DataLoader.

    - Se mueve el modelo al dispositivo especificado.
    - Se establece en modo evaluación.
    - Se calculan las predicciones y se comparan con las etiquetas reales.

    Args:
        model (torch.nn.Module): Modelo PyTorch a evaluar.
        dataloader (DataLoader): DataLoader con el conjunto de datos de prueba.
        device (str, optional): Dispositivo en el que se evaluará el modelo ('cpu' o 'cuda'). Por defecto, 'cpu'.

    Returns:
        float: Precisión (accuracy) del modelo en el dataset dado.
    """
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():  # No calcular gradientes
        for texts, labels, lengths in dataloader:
            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts, lengths).squeeze()  # Obtener predicciones

            # Convertir salidas a etiquetas en {-1, 0, 1}
            preds = torch.zeros_like(outputs, dtype=torch.long)
            preds[outputs > 0.5] = 1
            preds[outputs < -0.5] = -1

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def train_torch_model(model: torch.nn.Module, train_dataloader: DataLoader,
                val_dataloader: DataLoader, criterion: torch.nn.Module,
                optimizer: optim.Optimizer, epochs: int,
                print_every: int, patience: int,
                device: str = 'cpu') -> Tuple[Dict[int, float],Dict[int, float]]:
    """
    Train and validate the logistic regression model.

    Args:
        model (torch.nn.Module): An instance of the model to be trained.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        learning_rate (float): The learning rate for the optimizer.
        criterion (nn.Module): Loss function to use for training.
        optimizer (optim.Optimizer): Optimizer to use for training.
        epochs (int): The number of epochs to train the model.
        print_every (int): Frequency of epochs to print training and validation loss.
        patience (int): The number of epochs to wait for improvement on the validation loss before stopping training early.
        device (str): device where to train the model.

    Returns:
        Tuple[Dict[int, float],Dict[int, float]]: Dictionary of accuracies at each `print_every` interval for the training and validation datasets.
    """
    # TODO: Initialize dictionaries to store training and validation accuracies
    train_accuracies: Dict[int, float] = {}
    val_accuracies: Dict[int, float] = {}

    # TODO: Initialize variables for Early Stopping
    best_loss: float = float('inf')
    epochs_no_improve: int = 0

    # TODO: Move the model to the specified device (CPU or GPU)
    model.to(device)

    # TODO: Implement the training loop over the specified number of epochs
    for epoch in range(epochs):
        # TODO: Set the model to training mode
        model.train()
        total_loss: int = 0.0

        # TODO: Implement the loop for training over each batch in the training dataloader
        for features, labels, text_len in train_dataloader:
            pass
            # TODO: Move features and labels to the specified device
            features, labels = features.to(device), labels.to(device)

            # TODO: Clear the gradients
            optimizer.zero_grad()

            # TODO: Forward pass (compute the model output)
            outputs = model(features, text_len)

            # TODO: Compute the loss
            loss = criterion(outputs.squeeze(), labels.float())

            # TODO: Backward pass (compute the gradients)
            loss.backward()

            # TODO: Update model parameters
            optimizer.step()

            # TODO: Accumulate the loss
            total_loss += loss.item()

        # TODO: Implement the evaluation phase
        model.eval()
        val_loss: int = 0.0
        with torch.no_grad():
            # TODO: Loop over the validation dataloader
            for features, labels, text_len in val_dataloader:
                pass
                # TODO: Move features and labels to the specified device
                features, labels = features.to(device), labels.to(device)

                # TODO: Forward pass (compute the model output)
                outputs = model(features, text_len)

                # TODO: Compute the loss
                loss = criterion(outputs.squeeze(), labels.float())

                # TODO: Accumulate validation loss
                val_loss += loss.item()

        # TODO: Print training and validation results every 'print_every' epochs
        if epoch % print_every == 0 or epoch == epochs - 1:
            pass
            # TODO: Calculate training and validation accuracy
            train_acc = calculate_accuracy_SA(model, train_dataloader, device=device)
            val_acc = calculate_accuracy_SA(model, val_dataloader, device=device)

            # TODO: Store accuracies
            train_accuracies[epoch] = train_acc
            val_accuracies[epoch] = val_acc

            # TODO: Calculate and print average losses and accuracies
            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {total_loss / len(train_dataloader):.4f} | "
                  f"Val Loss: {val_loss / len(val_dataloader):.4f} | "
                  f"Train Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}%")

        # TODO: Implement Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    return train_accuracies, val_accuracies
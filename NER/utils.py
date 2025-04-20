import torch
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import defaultdict

# funciones y clases propias
from NER.LSTM import BiLSTM

batch_size: int = 32
epochs: int = 50
print_every: int = 5
patience: int = 2
learning_rate: float = 0.001
hidden_dim: int = 256
num_layers: int = 2
dropout_p: float = 0.5
bidirectional: bool = True
embedding_dim = 300

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def calculate_confusion_matrix_NER(
    model: torch.nn.Module, dataloader: DataLoader, tag2idx, device: str = "cpu"
) -> None:
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
                pred_seq = predicted_tags[i][: lengths[i]].cpu().tolist()
                label_seq = tags[i][: lengths[i]].cpu().tolist()

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
    """Calculates class weights for a Named Entity Recognition (NER) task using scikit-learn's `compute_class_weight` function.
    This function computes the weights for each class based on the frequency of their occurrence in the dataset.
    The weights are then returned as a PyTorch tensor, where each index corresponds to a class, and the padding index (if present) is assigned a weight of 0.
    Args:
        tag2idx (dict): A dictionary mapping NER tags to their corresponding indices.
                        For example, {'O': 0, 'B-PER': 1, 'I-PER': 2, ...}.
        dataset (Dataset): A dataset object containing a `ner_tags` attribute, which is a list of lists where each sublist
                           contains the NER tag indices for a sentence.
    Returns:
        torch.Tensor: A tensor of class weights where the index corresponds to the class index.
                      Padding index (if present) is assigned a weight of 0."""

    all_tags = []

    for tags in dataset.ner_tags:
        all_tags.extend(tags)

    # Eliminar las etiquetas de padding (-1)
    all_tags = [tag for tag in all_tags if tag != -1]

    classes = list(tag2idx.values())
    classes = [c for c in classes if c != -1]  # quitamos el índice del padding

    # Calcular los pesos
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=all_tags
    )

    # Crear un tensor de pesos donde el índice corresponde a cada clase
    weight_tensor = torch.zeros(len(tag2idx), dtype=torch.float)

    for i, cls in enumerate(classes):
        weight_tensor[cls] = class_weights[i]

    # Asegúrate de que el índice de padding tenga peso 0
    if -1 in tag2idx.values():
        pad_idx = [k for k, v in tag2idx.items() if v == -1][0]
        weight_tensor[tag2idx[pad_idx]] = 0.0

    return weight_tensor


def calculate_accuracy_per_tag(
    model: torch.nn.Module, dataloader: DataLoader, tag2idx, device: str = "cpu"
) -> dict:
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
                pred_seq = predicted_tags[i][: lengths[i]].cpu().tolist()
                label_seq = tags[i][: lengths[i]].cpu().tolist()

                for pred_tag, true_tag in zip(pred_seq, label_seq):
                    true_tag_name = idx2tag[true_tag]

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
            "total": total[tag],
        }

    return accuracy_per_tag


def calculate_accuracy_NER(
    model: torch.nn.Module, dataloader: DataLoader, device: str = "cpu"
) -> float:
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
                mask[i, : lengths[i]] = 1

            correct += ((predicted_tags == tags) & mask).sum().item()
            total += mask.sum().item()

    return correct / total if total > 0 else 0.0


def train_torch_model(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    print_every: int,
    patience: int,
    full_train_dataset: Dataset,
    device: str = "cpu",
) -> dict[int, float]:
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
        full_train_dataset (Dataset): Full training dataset, used for
                            saving metadata like vocabulary and tag mappings.
        device (str): Device to train on ('cpu' or 'cuda').

    Returns:
        Dict[int, float]: Dictionary with training and validation accuracies.
    """
    train_accuracies = {}
    val_accuracies = {}
    best_val_loss = float("inf")
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

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]), tags.reshape(-1)
            )  # Flatten both
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
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]), tags.reshape(-1)
                )
                val_loss += loss.item()

        # Logging
        if epoch % print_every == 0 or epoch == epochs - 1:
            train_acc = calculate_accuracy_NER(model, train_dataloader, device)
            val_acc = calculate_accuracy_NER(model, val_dataloader, device)
            train_accuracies[epoch] = train_acc
            val_accuracies[epoch] = val_acc
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss / len(train_dataloader):.4f} | "
                f"Val Loss: {val_loss / len(val_dataloader):.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_model(
                model,
                optimizer,
                epoch,
                model_path="model_NER.pth",
                vocab_size=len(full_train_dataset.word2idx),
                embedding_dim=embedding_dim,
                tagset_size=len(full_train_dataset.tag2idx),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout_p=dropout_p,
                pad_idx=full_train_dataset.pad_idx,
                word2idx=full_train_dataset.word2idx,
                tag2idx=full_train_dataset.tag2idx,
            )
        else:
            epochs_no_improve += 1
            patience_left = patience - epochs_no_improve
            print(f"Validation loss did not improve. Patience left: {patience_left}")
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    return train_accuracies, val_accuracies


def save_model(
    model: torch.nn.Module,
    optimizer,
    epoch,
    model_path: str = "model_NER.pth",
    vocab_size=None,
    embedding_dim=None,
    tagset_size=None,
    hidden_dim=None,
    num_layers=None,
    dropout_p=None,
    pad_idx=None,
    word2idx=None,
    tag2idx=None,
):
    """
    Saves the state of a PyTorch model along with its optimizer and additional metadata.
    Args:
        model (torch.nn.Module): The PyTorch model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer associated with the model.
        epoch (int): The current epoch number, to be saved for resuming training.
        model_path (str, optional): The filename for saving the model. Defaults to "model_NER.pth".
        vocab_size (int, optional): The size of the vocabulary used in the model. Defaults to None.
        embedding_dim (int, optional): The dimensionality of the embedding layer. Defaults to None.
        tagset_size (int, optional): The number of tags in the tagset. Defaults to None.
        hidden_dim (int, optional): The size of the hidden layer in the model. Defaults to None.
        num_layers (int, optional): The number of layers in the model. Defaults to None.
        dropout_p (float, optional): The dropout probability used in the model. Defaults to None.
        pad_idx (int, optional): The index used for padding in the vocabulary. Defaults to None.
        word2idx (dict, optional): A dictionary mapping words to their corresponding indices. Defaults to None.
        tag2idx (dict, optional): A dictionary mapping tags to their corresponding indices. Defaults to None.
    Saves:
        A dictionary containing:
            - 'epoch': The current epoch number.
            - 'model_state_dict': The state dictionary of the model.
            - 'optimizer_state_dict': The state dictionary of the optimizer.
            - 'vocab_size': The size of the vocabulary.
            - 'embedding_dim': The dimensionality of the embedding layer.
            - 'tagset_size': The number of tags in the tagset.
            - 'hidden_dim': The size of the hidden layer.
            - 'num_layers': The number of layers in the model.
            - 'dropout_p': The dropout probability.
            - 'pad_idx': The padding index.
            - 'word2idx': The word-to-index mapping.
            - 'tag2idx': The tag-to-index mapping.
    The model and metadata are saved in the "saved_models" directory within the BASE_DIR.
    """

    models_path = os.path.join(BASE_DIR, "saved_models")
    os.makedirs(models_path, exist_ok=True)
    model_path = os.path.join(models_path, model_path)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "tagset_size": tagset_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout_p": dropout_p,
            "pad_idx": pad_idx,
            "word2idx": word2idx,
            "tag2idx": tag2idx,
        },
        model_path,
    )


def load_ner(model_path: str = "model_NER.pth", device: str = "cpu"):
    """
    Load a pre-trained Named Entity Recognition (NER) model from a checkpoint file.
    Args:
        model_path (str): The relative path to the model checkpoint file.
                          Defaults to "model_NER.pth".
        device (str): The device to load the model onto ("cpu" or "cuda").
                      Defaults to "cpu".
    Returns:
        tuple: A tuple containing:
            - model (BiLSTM): The loaded NER model.
            - word2idx (dict): A dictionary mapping words to their corresponding indices.
            - tag2idx (dict): A dictionary mapping tags to their corresponding indices.
            - pad_idx (int): The index used for padding in the model.
    Raises:
        FileNotFoundError: If the specified model checkpoint file does not exist.
        RuntimeError: If there is an issue loading the model state dictionary.
    Note:
        The function assumes that the checkpoint file contains the following keys:
        - 'vocab_size': Size of the vocabulary.
        - 'embedding_dim': Dimension of the word embeddings.
        - 'tagset_size': Number of unique tags in the NER task.
        - 'hidden_dim': Dimension of the hidden layers in the BiLSTM model.
        - 'num_layers': Number of layers in the BiLSTM model.
        - 'dropout_p': Dropout rate used during training.
        - 'pad_idx': Padding index used in the model.
        - 'model_state_dict': State dictionary of the trained model.
        - 'word2idx': Dictionary mapping words to indices.
        - 'tag2idx': Dictionary mapping tags to indices.
    """
    model_path = os.path.join("NER/saved_models", model_path)

    # Cargar el checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Crear el modelo con la misma configuración
    model = BiLSTM(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        tagset_size=checkpoint["tagset_size"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        dropout_rate=checkpoint["dropout_p"],
        pad_idx=checkpoint["pad_idx"],
    ).to(device)

    # Cargar los parámetros del modelo
    model.load_state_dict(checkpoint["model_state_dict"])

    # Imprimir que el modelo se ha cargado
    print(f"Modelo NER cargado desde {model_path}")

    # Devolver el modelo y la metadata necesaria
    return model, checkpoint["word2idx"], checkpoint["tag2idx"], checkpoint["pad_idx"]


def evaluate(
    rnn_model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    device,
    full_train_dataset,
):
    """
    Evaluates the performance of a Named Entity Recognition (NER) model on training, validation,
    and test datasets. Additionally, computes accuracy per tag and generates a confusion matrix.
    Args:
        rnn_model (torch.nn.Module): The trained RNN-based NER model to evaluate.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        full_train_dataset (Dataset): The full training dataset containing tag-to-index mapping (tag2idx).
    Returns:
        None: This function prints the evaluation results, including:
            - Training, validation, and test accuracy.
            - Accuracy per tag for the test dataset.
            - Confusion matrix for the test dataset.
    """

    # Final evaluation on train, validation, and test datasets
    train_acc = calculate_accuracy_NER(rnn_model, train_dataloader, device=device)
    val_acc = calculate_accuracy_NER(rnn_model, val_dataloader, device=device)
    test_acc = calculate_accuracy_NER(rnn_model, test_dataloader, device=device)

    # Print results
    print(f"\nNER Model - Training Accuracy: {train_acc:.4f}")
    print(f"NER Model - Validation Accuracy: {val_acc:.4f}")
    print(f"NER Model - Test Accuracy: {test_acc:.4f}")

    tag_accuracy = calculate_accuracy_per_tag(
        rnn_model, test_dataloader, full_train_dataset.tag2idx, device
    )

    print("\nAccuracy por etiqueta (NER):")
    for tag in sorted(tag_accuracy.keys(), key=lambda t: full_train_dataset.tag2idx[t]):
        info = tag_accuracy[tag]
        print(
            f"{str(full_train_dataset.tag2idx[tag]):8s} → {info['accuracy']:.4f}  (Correctas: {info['correct']}, Totales: {info['total']})"
        )

    # Después de entrenar el modelo y realizar la evaluación
    calculate_confusion_matrix_NER(
        rnn_model, test_dataloader, full_train_dataset.tag2idx, device=device
    )

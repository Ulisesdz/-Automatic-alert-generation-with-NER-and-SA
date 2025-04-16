import torch
from torch.utils.data import DataLoader, random_split
from LSTM import BiLSTM  # Assuming your LSTM model is in LSTM.py
from utils import train_torch_model, load_word2vec, calculate_class_weights_sklearn, evaluate  
from datasets import NERWord2VecDataset, create_collate_fn


# Hyperparameters
batch_size: int = 64
epochs: int = 50
print_every: int = 5
patience: int = 10
learning_rate: float = 0.001
hidden_dim: int = 128 #16, 32, 263
num_layers: int = 3 #2
dropout_p: float = 0.5
bidirectional: bool = True
embedding_dim = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_csv = "../data/NER/train/conll2003_train.csv"
test_csv = "../data/NER/test/conll2003_test.csv"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_csv = "../data/NER/train/conll2003_train.csv"
    test_csv = "../data/NER/test/conll2003_test.csv"

    word2vec_model = load_word2vec()
    embedding_weights = torch.tensor(word2vec_model.vectors, dtype=torch.float32)
    print('embedding check')

    full_train_dataset = NERWord2VecDataset(train_csv, word2vec_model)

    # Split dataset into 80% train and 20% validation
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, 
                                              [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(42))

    # Load test dataset
    test_dataset = NERWord2VecDataset(test_csv, word2vec_model)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True, 
                                  collate_fn=create_collate_fn())
    
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                collate_fn=create_collate_fn())
    
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 collate_fn=create_collate_fn())
    print('dataloader check')
    # Create the pre-trained embeddings

    # Calculamos los weights
    weights = calculate_class_weights_sklearn(full_train_dataset.tag2idx,full_train_dataset)

    # Crear el modelo con los embeddings preentrenados
    rnn_model = BiLSTM(
            embedding_dim=embedding_dim,
            tagset_size=len(full_train_dataset.tag2idx), # o el tamaño real de tu conjunto de etiquetas
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_p,
            pretrained_embeddings=embedding_weights,       # el índice del token de padding
        ).to(device)                

    # Define loss function and optimizer
    # Modificar la función de pérdida para usar los pesos de clase
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=-1)
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=learning_rate)

    print("training")
    # Train the model
    train_accuracies, val_accuracies = train_torch_model(
        rnn_model, train_dataloader, val_dataloader, criterion,
        optimizer, epochs, print_every, patience, device=device
    )   

    evaluate(rnn_model, train_dataloader, val_dataloader, test_dataloader, device, full_train_dataset)
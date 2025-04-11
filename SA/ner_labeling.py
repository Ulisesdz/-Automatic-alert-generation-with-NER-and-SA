import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split

# funciones y clases propias
from datasets import Conll2003Dataset, CollateFn
from utils import load_model, load_word2vec

# -------- Configuración --------
model_path = "saved_models/model_SA.pth"
test_csv = "../data/SA+NER/test/conll2003_test_SA.csv"
word2vec_path = "models/word2vec-google-news-300.kv"
batch_size = 64
result_path = "../SA/SA+NER/results.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- Evaluación --------
if __name__ == "__main__":
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    # Cargar el modelo Word2Vec
    word2vec_model = load_word2vec()
    embedding_weights = torch.tensor(word2vec_model.vectors, dtype=torch.float32)

    # Cargar el dataset y preparar el DataLoader
    test_df = pd.read_csv(test_csv)

    # Columna 'sentence' para hacer la predicción
    test_sentences = test_df["sentence"].tolist()
    test_sentiment = test_df["sentiment"].tolist()

    # Convertir las oraciones en índices de Word2Vec
    test_dataset = Conll2003Dataset(test_csv, word2vec_model)
    test_size = len(test_dataset)  
    test_dataset, _ = random_split(test_dataset, [test_size, len(test_dataset) - test_size],
                                   generator=torch.Generator().manual_seed(42))

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=CollateFn(word2vec_model))

    # Modelo entrenado
    model = load_model(model_path, embedding_weights, device=device)
    model.eval() 

    predictions = []
    true_labels = []
    
    # Predecir el sentimiento para cada oración
    with torch.no_grad():
        for sentences, labels, lengths in test_dataloader:
            sentences = sentences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)  
            
            outputs = model(sentences, lengths)  
            predicted = torch.round(torch.sigmoid(outputs))  # Convertir a 0 o 1
            
            # Guardar las predicciones y las etiquetas 
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Calcular el porcentaje de coincidencias
    correct = sum([1 if p == t else 0 for p, t in zip(predictions, true_labels)])
    total = len(true_labels)
    accuracy = correct / total * 100
    print(f"Porcentaje de coincidencias: {accuracy:.2f}%")

    # Guardar las predicciones
    predicted_sentiments = [int(pred.item()) for pred in predictions]  # Usamos .item() para obtener el valor escalar

    # Añadir las predicciones al DataFrame
    test_df['predicted_sentiment'] = predicted_sentiments
    test_df['sentiment'] = test_df['sentiment'].apply(lambda x: 1 if x == 1 else 0)

    # Guardar el DataFrame con las predicciones
    test_df[['sentence', 'sentiment', 'predicted_sentiment']].to_csv(result_path, index=False)
    print(f"Resultados guardados en {result_path}")
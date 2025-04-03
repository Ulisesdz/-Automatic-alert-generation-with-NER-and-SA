import os
import re
import pandas as pd
from typing import List

# Diccionario para expandir contracciones comunes en inglés
CONTRACTIONS = {
    "can't": "can not", "won't": "will not", "don't": "do not", "doesn't": "does not",
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "they're": "they are", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not",
    "mustn't": "must not", "let's": "let us", "that's": "that is", "what's": "what is",
    "here's": "here is", "there's": "there is", "who's": "who is", "how's": "how is",
    "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would", 
    "we'd": "we would", "they'd": "they would", "i'll": "i will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will", "we'll": "we will", "they'll": "they will",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have"
}

# Diccionario de emociones clasificadas
EMOTION_CATEGORIES = {
    "yawn": "BAD_EMOTION", "sad": "BAD_EMOTION", "cry": "BAD_EMOTION",
    "angry": "BAD_EMOTION", "frustrated": "BAD_EMOTION", "annoyed": "BAD_EMOTION",
    "jealous": "BAD_EMOTION", "mad": "BAD_EMOTION", "upset": "BAD_EMOTION",
    "happy": "GOOD_EMOTION", "excited": "GOOD_EMOTION", "love": "GOOD_EMOTION",
    "joy": "GOOD_EMOTION", "smile": "GOOD_EMOTION", "laugh": "GOOD_EMOTION",
    "grateful": "GOOD_EMOTION", "satisfied": "GOOD_EMOTION", "relieved": "GOOD_EMOTION",
    "hug": "GOOD_EMOTION", "kiss": "GOOD_EMOTION", "swoon": "GOOD_EMOTION",
}

def expand_contractions(text: str) -> str:
    """Expande contracciones comunes en inglés."""
    words = text.split()
    expanded_words = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    return " ".join(expanded_words)

def normalize_repeated_chars(word: str) -> str:
    """Reduce caracteres repetidos en una palabra (ej. allll → all, soooo → so)."""
    return re.sub(r"(.)\1{2,}", r"\1", word)

def replace_emotions(text: str) -> str:
    """
    Reemplaza palabras entre ** con etiquetas de emociones categorizadas.
    Ejemplo: **yawn** → <BAD_EMOTION>, **hug** → <GOOD_EMOTION>, **unknown** → <EMOTION>.
    """
    def emotion_replacer(match):
        emotion = match.group(1).lower()  # Convertir a minúsculas
        category = EMOTION_CATEGORIES.get(emotion, "EMOTION")
        return f"<{category}>"

    return re.sub(r"\*\*(.*?)\*\*", emotion_replacer, text)

def clean_text(text: str) -> str:
    """
    Aplica limpieza adicional:
    - Elimina caracteres raros como `- , . " ...`
    - Tokeniza `!` y `?` como tokens separados.
    - Reemplaza emojis comunes con `<EMOJI>` (se puede mejorar con un diccionario).
    - Limpia espacios en blanco extra.
    """
    text = re.sub(r"[,.\"…]", "", text)  # Eliminar caracteres innecesarios
    text = re.sub(r"([!?])", r" \1 ", text)  # Separar ! y ?
    text = re.sub(r"[\U00010000-\U0010ffff]", "<EMOJI>", text)  # Reemplazar emojis con <EMOJI>
    text = re.sub(r"\s+", " ", text).strip()  # Limpiar espacios extra
    return text

def tokenize_tweet(tweet: str) -> List[str]:
    """
    Tokeniza y normaliza un tweet:
    - Convierte a minúsculas.
    - Sustituye menciones (@usuario) con <USER>.
    - Sustituye URLs con <URL>.
    - Sustituye hashtags con <HASHTAG>.
    - Expande contracciones (ej. can't → can not).
    - Normaliza caracteres repetidos (ej. alllll → all).
    - Sustituye emociones detectadas con etiquetas <GOOD_EMOTION>, <BAD_EMOTION> o <EMOTION>.
    - Aplica limpieza de caracteres raros y tokeniza signos de puntuación clave.
    """
    tweet = tweet.lower()  # Convertir a minúsculas
    tweet = re.sub(r"@\w+", "<USER>", tweet)  # Reemplazar menciones de usuario
    tweet = re.sub(r"http\S+|www\S+", "<URL>", tweet)  # Reemplazar URLs
    tweet = re.sub(r"[#]\w+", "<HASHTAG>", tweet)  # Reemplazar hashtags

    tweet = expand_contractions(tweet)  # Expandir contracciones
    tweet = replace_emotions(tweet)  # Reemplazar emociones entre **
    tweet = clean_text(tweet)  # Aplicar limpieza extra

    tokens = tweet.split()  # Tokenizar dividiendo en palabras
    tokens = [normalize_repeated_chars(word) for word in tokens]  # Normalizar caracteres repetidos

    return tokens

def process_sentiment140():
    file_path = "..raw_data/sentiment140/training.1600000.processed.noemoticon.csv"
    if not os.path.exists(file_path):
        print("Sentiment140 dataset not found.")
        return

    # Crear carpetas de salida si no existen
    os.makedirs("data/SA/train", exist_ok=True)
    os.makedirs("data/SA/test", exist_ok=True)

    # Cargar dataset
    df = pd.read_csv(file_path, encoding="latin1", header=None, names=["target", "ids", "date", "flag", "user", "text"])

    # Convertir etiquetas de sentimiento a texto
    df["target"] = df["target"].replace({0: -1, 2: 0, 4: 1})

    # Aplicar tokenización y limpieza
    df["text"] = df["text"].apply(lambda x: " ".join(tokenize_tweet(x)))

    # Dividir en train (80%) y test (20%)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # Guardar en archivos CSV
    train_df.to_csv("..data/SA/train/sentiment140_train.csv", index=False)
    test_df.to_csv("..data/SA/test/sentiment140_test.csv", index=False)

    print("Processed Sentiment140 and saved train/test splits.")

if __name__ == "__main__":
    process_sentiment140()

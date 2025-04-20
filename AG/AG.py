import pandas as pd
import ast
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# PATH A LOS RESULTADOS NER + SA
path_input =  os.path.join("..","ner_sa_output", "ner_sa_output.csv")


# Modelo para prompting
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
OUTPUT_CSV = "generated_alerts.csv"
TOKEN = None


# Promt para alert generation
base_prompt = """<INSTRUCTIONS_FOR_YOU>

You are an alert generation system. In this conversation, you will receive the following information:

<TEXT>
This is the full input text that has been analyzed.
</TEXT>

<ENTITIES>
These are the named entities found in the text, with their types in parentheses.
</ENTITIES>

<SENTIMENT>
This is the overall sentiment detected in the text (positive, negative, or neutral).
</SENTIMENT>

Your task is to generate an alert in a single sentence. The alert must always mention the sentiment and entities (if provided) and summarize the input.
The tone should be clear and concise, as if it were a notification from an intelligent monitoring system. Do not add any information that is not explicitly present in the input.

Your output should be one natural-sounding sentence in English, you must always provide an alert.

</INSTRUCTIONS_FOR_YOU>

Now the conversation begins:

<TEXT>
{sentence}
</TEXT>

<ENTITIES>
{entities}
</ENTITIES>

<SENTIMENT>
{sentiment}
</SENTIMENT>

Please generate an alert:
"""


# Generación de alertas
def clean_entities(entities_str):
    """Convertir lista de str a tipo list"""
    try:
        return ast.literal_eval(entities_str)
    except:
        return []


def generar_alertas(df, pipe):
    alerts = []
    for _, row in df.iterrows():
        sentence = row["caption"]
        entities = row["entities"]
        sentiment = row["sentiment"]

        # Normaliza entidades
        if isinstance(entities, str):
            parsed = clean_entities(entities)
            if isinstance(parsed, list):
                entities = ", ".join(parsed)
            else:
                entities = str(entities)
        print("entidades", entities)
        # Construcción de prompt
        prompt = base_prompt.format(
            sentence=sentence,
            entities=entities,
            sentiment=sentiment
        )
        print("prompt",prompt)
        # Generación con el modelo
        result = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)[0]["generated_text"]
        print("resultado", result)
        alert = result.strip().split("\n")[-2].strip()  # Obtener solo la alerta entre los mensajes de <ALERT> </ALERT>
        alerts.append(alert)
    return alerts

def main():
    print("Cargando modelo y tokenizer de Llama...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,  token=TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.float32, token=TOKEN)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print(f"Leyendo CSV desde: {path_input}")
    df = pd.read_csv(path_input)

    print("Generando alertas...")
    df["alert"] = generar_alertas(df, pipe)

    print(f"Guardando en: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()
# Automatic Alert Generation with NER and SA

This project involves the development of an **automatic alert generation system** from news articles and social media posts. The system leverages **Named Entity Recognition (NER)** and **Sentiment Analysis (SA)** techniques, along with **image captioning**, to produce contextualized alerts. 

---

## System Overview

Given a **textual input** (e.g., from an article or tweet) and a related **image**, the system follows this pipeline:

1. **Image Captioning**  
   A pretrained BLIP model generates a textual caption from the image.

2. **Text Fusion**  
   The generated caption is concatenated with the original input text to enrich context.

3. **Named Entity Recognition (NER)**  
   The fused text is processed using a NER model to extract relevant entities (persons, locations, organizations, etc.).

4. **Sentiment Analysis (SA)**  
   A pretrained SA model classifies the overall sentiment as **positive**, **neutral**, or **negative**.

5. **Alert Generation**  
   The final step takes the entities, sentiment, and text as input and generates a concise one-sentence alert using a language generation model.

---

## Repository Structure

```bash
Automatic-Alert-Generation-WITH-NER-AND-SA/

├── AG/                                  # Alert Generation
│   ├── AG.py                            # Generate Alert from ner_sa_output.csv
│   ├── inference_pipeline.py            # Generate SA, NER and IC from captions_input.csv 
│   ├── generated_alerts.csv             # Generated Alert Result
│   └── ner_sa_output.csv                # Results of inference_pipeline.py 

├── DataPreprocess/                      # Data preprocessing scripts
│   ├── download_datasets.py
│   ├── load_ner_dataset.py
│   ├── SA_preprocess.py
│   └── sentiment_classifier.py

├── image_captions/                      # Image Captioning
├── ├── IMAGES/                          # Images for AG
│   ├── run_blip_captioning.py           # Generate captions for AG
│   ├── captions_example.csv             # Results test BLIP Model
│   └── captions_input.csv               # Script to try BLIP model

├── SA/                                  # Sentiment Analysis
│   ├── SA+NER/                          # Results of ner_labeling_neutral.py 
│   ├── saved_models/                    # Trained Model
│   ├── datasets.py                      # Datasets structure
│   ├── evaluate.py                      # Evaluate SA Model
│   ├── sentiment_analysis.py            # Train SA Model         
│   ├── LSTM.py                          # BiLSTM+Att Structure
│   ├── utils.py                         # Utils Functions
│   └── ner_labeling_neutral.py          # Compare SA Model with Pretrained Model

├── NER/                                 # Named Entity Recognition
│   ├── saved_models/                    # Trained Model
│   ├── datasets.py                      # Datasets structure
│   ├── evaluate.py                      # Evaluate NER Model
│   ├── NER.py                           # Train NER Model   
│   ├── LSTM.py                          # BiLSTM Structure
│   └── utils.py                         # Utils Functions
```

## How to Run the Project

### Clone the Repository
```bash
git clone https://github.com/your-username/Automatic-Alert-Generation-WITH-NER-AND-SA.git
cd Automatic-Alert-Generation-WITH-NER-AND-SA
```
### Install Required Dependencies
```bash
pip install -r requirements.txt
```
### Data Downloading and Preprocessing
This step will create the raw_data/ and data/ folders and download/process the necessary datasets.
```bash
python -m DataPreprocess.download_datasets
python -m DataPreprocess.load_ner_dataset
python -m DataPreprocess.SA_preprocess
python -m DataPreprocess.sentiment_classifier
```
### Train NER and SA Models
To modify hyperparameters, edit:
- NER/NER.py and NER/utils.py for the NER model.
- SA/sentiment_analysis.py and SA/utils.py for the SA model.
```bash
python -m NER.NER
python -m SA.sentiment_analysis
```
### Evaluate Trained Models
```bash
python -m NER.evaluate
python -m SA.evaluate
python -m SA.ner_labeling_neutral
```
### Generate Alerts from Input Data
1. Add your image(s) to the image_captions/IMAGES/ folder.
2. Add a new row to image_captions/captions_input.csv with the format:
```css
image_name,caption
```
Model Configurations
- The image captioning model used is blip-image-captioning-base. Change the variable MODEL_ID in AG/inference_pipeline.py to use a different one.
- The alert generation model used is Llama-2-7b. Change the variable MODEL_ID in AG/AG.py to switch models.
```bash
python -m AG.inference_pipeline
python -m AG.AG
```
Another option if the downloading and prediction of the model takes too long it to upload the ner_sa_output.csv and the AG.py to colab, which is faster. No adaptations to the code have to be done.

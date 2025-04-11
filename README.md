# Automatic Alert Generation with NER and SA

This project involves the development of an **automatic alert generation system** from news articles and social media posts. The system leverages **Named Entity Recognition (NER)** and **Sentiment Analysis (SA)** techniques, along with **image captioning**, to produce contextualized alerts. 

These alerts are designed to assist in:

- Reputation tracking  
- Economic updates  
- Geopolitical risk detection  
- Social media monitoring  
- News summarization  

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


## Repository Structure

Automatic-Alert-Generation-WITH-NER-AND-SA/
│
├── AG/
│   ├── AlertGeneration_model.ipynb      # Alert generation pipeline
│   └── predictions_test.csv             # Output example
│
├── Bibliography/                        # References and research materials
│
├── data/                                # Preprocessed input data
│   ├── IMAGES/                          # Image inputs
│   ├── NER/                             # NER output samples
│   ├── SA/                              # Sentiment output samples
│   └── SA+NER/                          # Joint outputs
│
├── DataPreprocess/                      # Data preprocessing scripts
│   ├── download_datasets.py
│   ├── load_ner_dataset.py
│   ├── preprocess_images.py
│   ├── SA_preprocess.py
│   └── sentiment_classifier.py
│
├── Documentation/                      # Reports
│
├── image_captions/
│   ├── captions_output.csv              # Generated captions
│   └── run_blip_captioning.py           # Script to run BLIP model
│
├── raw_data/                            # Raw datasets
│   ├── conll2003/
│   ├── flickr30k/
│   └── sentiment140/
│
├── SA/                                  # Sentiment analysis modules
│   ├── models/                          # Word2Vec
│   ├── SA+NER/                          # Results Compare Final model with Pretrained
│   ├── saved_models/                    # Trained Model Saved
│   ├── datasets.py                      
│   ├── evaluate.py                      
│   ├── LSTM.py                          
│   ├── ner_labeling.py                  # Compare Final model with Pretrained
│   └── ner_labeling_neutral.py          # Compare Final model with Pretrained


## How to Run the Project

### Clone the Repository
```bash
git clone https://github.com/your-username/Automatic-Alert-Generation-WITH-NER-AND-SA.git
cd Automatic-Alert-Generation-WITH-NER-AND-SA
```

Run the scripts in DataPreprocess/ to prepare datasets for NER and SA.
# EmoTox: Hybrid Toxicity Detection Using Emotion and Sarcasm Analysis

EmoTox is a hybrid deep learning system for detecting whether a text comment is **toxic** or **non-toxic**. It combines three main ideas:

1. **Emotion Classification**
2. **Sarcasm Detection**
3. **Final Toxicity Classification**

The main goal of this project is to improve toxicity detection by looking beyond toxic keywords. Instead of only checking if a comment contains bad words, EmoTox also considers the emotional tone and possible sarcastic meaning behind the text.

> **Important Note About the Live Demo:**  
> The live demo is mainly for showing the full machine learning pipeline, model loading, tokenizer handling, custom Keras layer usage, and Streamlit deployment. The model performed well during training and evaluation, but the live demo may not always classify every input correctly, especially very short or direct profanity-only comments. This project is best viewed as an academic deep learning prototype, not a production-ready moderation system.

---

## Project Overview

Toxicity detection is important for online platforms because harmful comments can affect users and communities. However, toxic language is not always direct.

Some comments may be toxic because of:

- direct insults
- sarcasm
- passive-aggressive wording
- emotional hostility
- indirect harmful meaning

# System Process
The full EmoTox process follows this pipeline:
Input Text Dataset
        ↓
Data Preprocessing
        ↓
Keras Tokenization
        ↓
Word2Vec Embedding Layer
        ↓
Train / Validation / Test Split
        ↓
Emotion Model + Sarcasm Model
        ↓
Concatenated Probability Vectors
        ↓
Meta-Classifier
        ↓
Final Toxicity Prediction

# Datasets Used
The system uses multiple datasets for different tasks:
1. Wikipedia Toxicity Dataset - This dataset is used for the main toxic and non-toxic classification task.
2. SARC Dataset - This dataset is used for sarcasm detection.
3. Augmented Emotion Dataset - This dataset is used for emotion classification.
These datasets are preprocessed and combined to create a shared vocabulary and embedding representation.

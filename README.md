# 🧠 EmoTox — Hybrid Toxicity Detection Using Emotion and Sarcasm Analysis

A hybrid deep learning toxicity detection system built using **TensorFlow**, **Keras**, **Word2Vec**, **CNN-LSTM**, **BiLSTM with Multi-Head Attention**, and **Streamlit**.

EmoTox classifies text comments as **toxic** or **non-toxic** by combining emotion classification, sarcasm detection, and a final toxicity meta-classifier.

Unlike basic keyword-based toxicity detectors, EmoTox does not only look for offensive words. It also analyzes the **emotional tone**, **sarcastic meaning**, and **contextual pattern** of a comment before making the final prediction.

🔗 **Live Demo:** Add your Streamlit demo link here

> **Important Note About the Live Demo:**  
> The live demo is mainly intended to demonstrate the full machine learning pipeline, model loading, tokenizer handling, custom Keras layer usage, and Streamlit deployment.
>
> The model performed well during training and evaluation, but the live demo may not always classify every input correctly, especially very short or direct profanity-only comments. This project is best viewed as an **academic deep learning prototype**, not a production-ready moderation system.

---

# 🚀 Features

### 🧠 Hybrid Toxicity Detection
- Classifies text comments as **toxic** or **non-toxic**
- Uses emotion and sarcasm as supporting signals
- Designed to detect more than obvious toxic keywords
- Helps identify contextual toxicity, passive-aggressive wording, and indirect harmful comments

### ❤️ Emotion Classification
- Uses a **CNN-LSTM** deep learning model
- Detects emotional patterns in text
- Helps identify emotional cues such as anger, sadness, fear, joy, and other emotion-related signals
- Produces an emotion probability vector used by the final toxicity classifier

### 😏 Sarcasm Detection
- Uses a **Bidirectional Long Short-Term Memory (BiLSTM)** model
- Includes **Multi-Head Attention** through a custom Keras layer
- Helps detect sarcastic or indirect negative meaning
- Useful for comments that appear positive on the surface but are actually hostile or insulting

### 🔗 Ensemble Learning
- Combines the outputs of the emotion model and sarcasm model
- Uses probability vectors as additional features
- Final toxicity prediction is based on multiple contextual signals
- Improves over simple keyword-based or lexicon-based toxicity detection

### 🧪 Keras-Based Deep Learning Pipeline
- Built using **TensorFlow** and **Keras**
- Uses saved `.keras` model files
- Loads a custom Keras layer called `SelfAttentionBlock`
- Uses a saved Keras tokenizer for inference consistency

### 🌐 Streamlit Web Demo
- Simple web interface for testing comments
- Loads trained model files
- Processes user input through the same model pipeline
- Displays toxicity confidence score and final classification

---

# 🧠 How It Works

```text
User enters a comment
        ↓
Text is cleaned and preprocessed
        ↓
Keras tokenizer converts words into token IDs
        ↓
Token IDs are padded to a fixed sequence length
        ↓
Word2Vec embedding representation is used by the trained model
        ↓
Emotion model extracts emotional meaning
        ↓
Sarcasm model extracts sarcastic/contextual meaning
        ↓
Probability outputs are combined
        ↓
Meta-classifier predicts final toxicity score
        ↓
System returns Toxic or Non-toxic
```

In simple terms, EmoTox first prepares the text, converts it into numbers, analyzes emotion and sarcasm, then uses those signals to decide whether the comment is toxic.

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | Model building, training, saving, and loading |
| Streamlit | Web app interface and deployment |
| Word2Vec | Word embedding representation |
| Gensim | Word2Vec model handling |
| CNN-LSTM | Emotion classification architecture |
| BiLSTM | Sarcasm sequence modeling |
| Multi-Head Attention | Focuses on important contextual words |
| NumPy | Numerical operations |
| Pandas | Dataset handling and analysis |
| Scikit-learn | Evaluation metrics and data splitting |
| NLTK | Text preprocessing support |
| GitHub | Version control |
| Git LFS | Storage for large model files |
| Streamlit Cloud | Deployment platform |

---

# 🏗️ Architecture

## Full System Pipeline

```text
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
```

---

## Dataset Pipeline

```text
Wikipedia Toxicity Dataset
SARC Sarcasm Dataset
Augmented Emotion Dataset
        ↓
Dataset Cleaning and Preprocessing
        ↓
Dataset Concatenation
        ↓
Unified Text Corpus
        ↓
Shared Vocabulary
        ↓
Word2Vec Embedding Representation
```

The system uses different datasets because toxicity, sarcasm, and emotion are different but related tasks.

The **toxicity dataset** teaches the model toxic and non-toxic patterns.  
The **sarcasm dataset** teaches the model indirect or sarcastic meaning.  
The **emotion dataset** teaches the model emotional tone in text.

Together, these datasets help EmoTox understand more context than a normal keyword-based toxicity detector.

---

## Text Preprocessing Pipeline

```text
Raw Text Comment
        ↓
Remove unnecessary characters
        ↓
Remove URLs
        ↓
Convert text to lowercase
        ↓
Tokenization
        ↓
Remove punctuation
        ↓
Lemmatization
        ↓
Cleaned Text
```

Example:

```text
Original Comment:
Wow!!! You are SO stupid 😂 https://example.com

After Preprocessing:
wow you are so stupid
```

Preprocessing removes noise from the text so the model can focus on the actual words and patterns.

---

## Keras Tokenizer Pipeline

```text
Cleaned Text
        ↓
Split into words
        ↓
Convert each word into an integer ID
        ↓
Create token sequence
        ↓
Pad sequence to fixed length
```

Example:

```text
Text:
you are stupid

Token IDs:
[7, 24, 501]
```

The Keras tokenizer is important because deep learning models do not understand raw text. They only understand numbers.

For deployment, the tokenizer is saved as:

```text
models/tokenizer.pkl
```

The Streamlit app must use the same tokenizer used during training. If the tokenizer is different, the model may receive the wrong word IDs and produce unreliable predictions.

---

## Word2Vec Embedding Pipeline

```text
Token ID
        ↓
Look up word in embedding matrix
        ↓
Retrieve Word2Vec vector
        ↓
Pass vector sequence into model
```

Example:

```text
tokenizer.word_index["stupid"] = 501

embedding_matrix[501] = Word2Vec vector for "stupid"
```

A normal token ID only represents a word number. Word2Vec represents the meaning of the word as a vector.

Example:

```text
stupid → [0.12, -0.44, 0.03, ...]
```

These vectors help the model understand relationships between words. Words with similar meanings can have similar vector representations.

The saved Word2Vec file is:

```text
models/unified_word2vec_vectors.kv
```

---

# 📚 Datasets Used

### Wikipedia Toxicity Dataset
- Used for toxic and non-toxic classification
- Provides the main labels for toxicity detection
- Helps the final classifier learn harmful and non-harmful comment patterns

### SARC Dataset
- Used for sarcasm detection
- Helps the model identify indirect or sarcastic meaning
- Useful for comments where the surface meaning is different from the actual intent

### Augmented Emotion Dataset
- Used for emotion classification
- Helps the model detect emotional tone
- Supports the final toxicity model by providing emotion-based signals

---

# ❤️ Emotion Classification Model

The emotion model uses a **CNN-LSTM architecture**.

Its job is to detect the emotional tone of a comment. This matters because toxic comments often contain emotional signals such as anger, sadness, fear, or hostility.

## Emotion Model Architecture

```text
Word2Vec Embeddings
        ↓
Spatial Dropout
        ↓
LSTM
        ↓
1D Convolution
        ↓
Dropout
        ↓
LSTM
        ↓
1D Convolution
        ↓
Dropout
        ↓
Global Max Pooling
        ↓
Softmax Output
        ↓
Emotion Probability Vector
```

## Why CNN?

A **Convolutional Neural Network (CNN)** helps detect short word patterns or phrase-level features.

Example:

```text
you are stupid
```

The CNN can learn that this phrase may contain hostile or negative meaning.

## Why LSTM?

A **Long Short-Term Memory (LSTM)** layer understands word order and sequence.

Example:

```text
I am not happy with you
```

The meaning depends on the order of the words. LSTM helps the model understand this sequence.

## Emotion Output

The emotion model produces an emotion probability vector.

Example:

```text
anger: 0.70
joy: 0.02
sadness: 0.15
fear: 0.10
surprise: 0.03
```

This output becomes one of the inputs used by the final toxicity classifier.

---

# 😏 Sarcasm Detection Model

The sarcasm model uses a **Bidirectional Long Short-Term Memory (BiLSTM)** architecture with **Multi-Head Attention**.

Its job is to detect whether a comment contains sarcastic or indirect meaning.

Example:

```text
Great job, you ruined everything again.
```

The word **“great”** looks positive, but the full sentence is sarcastic and negative.

## Sarcasm Model Architecture

```text
Word2Vec Embeddings
        ↓
Bidirectional LSTM
        ↓
Multi-Head Attention
        ↓
Flatten
        ↓
Linear Layer
        ↓
Dropout
        ↓
Sigmoid Output
        ↓
Sarcasm Probability
```

## Why BiLSTM?

A normal LSTM reads text in one direction. A **BiLSTM** reads text in both directions.

```text
Forward:  word 1 → word 2 → word 3
Backward: word 3 → word 2 → word 1
```

This helps the model understand the context before and after each word.

## Why Multi-Head Attention?

**Multi-Head Attention** helps the model focus on important words or phrases that may signal sarcasm.

Example:

```text
Yeah, because that was such a smart idea.
```

The model may focus on phrases such as:

```text
yeah
such a smart idea
```

These phrases may signal sarcasm depending on the full sentence.

---

# 🧩 Custom Keras Layer: SelfAttentionBlock

The sarcasm model uses a custom Keras layer called `SelfAttentionBlock`.

This layer contains:

- `MultiHeadAttention`
- `LayerNormalization`
- residual connection

Simplified logic:

```python
attn_out = self.attn(x, x)
output = self.norm(x + attn_out)
```

## What This Means

The layer first applies attention to the input sequence. Then it adds the original input back to the attention output. This is called a **residual connection**.

After that, it normalizes the result using `LayerNormalization`.

This helps stabilize training and allows the model to keep both the original sequence information and the attention-enhanced information.

## Why This Matters in Deployment

Because `SelfAttentionBlock` is a custom layer, Keras cannot load the saved sarcasm model unless the same class is defined in `app.py`.

That is why the Streamlit app includes the custom layer code before loading the `.keras` models.

---

# 🔗 Ensemble Learning

EmoTox uses ensemble learning by combining the outputs of the emotion model and sarcasm model.

```text
Emotion Probability Vector
        +
Sarcasm Probability
        ↓
Concatenated Probability Vector
        ↓
Meta-Classifier
```

Instead of only asking:

```text
Does this comment contain toxic words?
```

The system also asks:

```text
What emotion does this comment express?
Is the comment sarcastic?
Do emotion and sarcasm together suggest toxicity?
```

This makes the system more context-aware than a simple keyword-based toxicity detector.

---

# 🧾 Meta-Classifier

The meta-classifier receives the combined probability outputs from the emotion and sarcasm models.

## Meta-Classifier Architecture

```text
Concatenated Probability Vectors
        ↓
Linear Layer
        ↓
ReLU Activation
        ↓
Sigmoid Output
        ↓
Toxicity Confidence Score
```

## Layer Explanation

### Linear Layer
Learns how to combine the emotion and sarcasm outputs.

### ReLU Activation
Helps the model learn more complex patterns.

### Sigmoid Output
Outputs a score between `0` and `1`.

Example:

```text
0.91 = high toxicity confidence
0.10 = low toxicity confidence
```

Final decision:

```text
score >= threshold → Toxic
score < threshold  → Non-toxic
```

---

# 🌐 Streamlit App Process

The Streamlit app is a simple web demo for using the trained model.

```text
User enters a comment
        ↓
Text is cleaned
        ↓
Text is tokenized
        ↓
Token IDs are padded to length 50
        ↓
The Keras model predicts toxicity confidence
        ↓
The app displays Toxic or Non-toxic
```

The app loads the following components:

```text
tokenizer.pkl
toxicity_classification_model.keras
SelfAttentionBlock custom layer
```

Input text is converted into token IDs using:

```python
sequence = tokenizer.texts_to_sequences(tokens)
```

Then it is padded using:

```python
padded_sequence = pad_sequences(
    sequence,
    maxlen=50,
    padding="post",
    truncating="post"
)
```

The toxicity model is called using:

```python
toxicity_model.predict([X_val, X_val])
```

This is because the saved toxicity model expects two input branches.

---

# 📁 Project Structure

```text
EmoTox/
│
├── app.py
├── README.md
├── requirements.txt
├── Thesis_2.ipynb
├── .gitignore
├── .gitattributes
│
├── models/
│   ├── emotion_classification_model.keras
│   ├── sarcasm_classification_model.keras
│   ├── toxicity_classification_model.keras
│   ├── unified_word2vec_vectors.kv
│   └── tokenizer.pkl
│
├── data/
│   └── toxicity_annotations.tsv
│
└── results/
    ├── confusion_matrix_results_modified.csv
    ├── matches_prediction.csv
    ├── mismatches_prediction.csv
    └── toxicity_test_set_with_predictions_modified.csv
```

---

# 📦 Model Files

| File | Purpose |
|---|---|
| `emotion_classification_model.keras` | Trained CNN-LSTM emotion classification model |
| `sarcasm_classification_model.keras` | Trained BiLSTM with Multi-Head Attention sarcasm detection model |
| `toxicity_classification_model.keras` | Final toxicity classification model |
| `unified_word2vec_vectors.kv` | Saved Word2Vec vectors |
| `tokenizer.pkl` | Saved Keras tokenizer used for inference |

---

# 📊 Results

Based on thesis evaluation, EmoTox outperformed the lexicon-based baseline.

| Model | Accuracy | F1-Score |
|---|---:|---:|
| Lexicon Baseline | 81.45% | 62.91% |
| EmoTox Proposed Model | 90.22% | 82.72% |

These results suggest that combining emotion and sarcasm features can improve toxicity detection compared to traditional lexicon-based methods.

---

# ⚠️ Live Demo Limitation

The live demo is working, but the predictions may not always behave as expected.

For example, very short profanity-only comments may sometimes be classified incorrectly.

This may happen because:

- The model learned from dataset patterns, not from a rule-based profanity list
- The toxicity dataset may contain imbalance
- The model may depend more on sentence context than single toxic words
- The deployed preprocessing may not perfectly reproduce every training step
- The project is an academic prototype, not a production moderation system

The live demo should be viewed as a demonstration of:

- Keras model loading
- custom Keras layer handling
- tokenizer usage
- deep learning inference
- Streamlit deployment
- end-to-end machine learning pipeline deployment

It should not be treated as a production-ready toxicity moderation API.

---

# 🧪 Engineering Concepts Used

- Natural Language Processing (NLP)
- Deep learning classification
- Word embeddings
- Word2Vec embedding matrix
- Keras Tokenizer
- CNN-LSTM architecture
- Bidirectional LSTM
- Multi-Head Attention
- Custom Keras layers
- Ensemble learning
- Meta-classification
- Model serialization and loading
- Streamlit deployment
- Git LFS for large model files

---

---

# 👤 Authors

## Ethan Lyle Cruz, Shan Hrvin Cabantugan, Sean Jonathan Estaya, Lance Owen Gulinao

- GitHub: https://github.com/Ethanol-15
- LinkedIn: https://www.linkedin.com/in/ethan-cruz-992730337/

---

# 📄 License

MIT License

---

Built with TensorFlow, Keras, Word2Vec, Streamlit, and a focus on context-aware toxicity detection.

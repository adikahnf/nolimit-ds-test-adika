# Tokopedia Product Reviews Sentiment Classification

A comprehensive end-to-end sentiment classification system for Indonesian e-commerce product reviews from Tokopedia, utilizing sentence transformers and machine learning techniques for accurate sentiment prediction.

## Project Description

This project implements a sentiment analysis pipeline for product reviews written in Indonesian (Bahasa Indonesia). The system classifies customer reviews into positive, negative, and neutral sentiments using Hugging Face transformers and FAISS for semantic search.

## Features : 
- **End-to-End Sentiment Classification Pipeline**: Complete workflow from data loading to model ready
- **Multilingual Sentence Transformers**: Uses `paraphrase-multilingual-MiniLM-L12-v2` for Indonesian text embeddings
- **Semantic Search Integration**: FAISS-powered similarity search for finding related reviews

## Dataset : 
**Source**: [Tokopedia Product Reviews - Kaggle](https://www.kaggle.com/datasets/farhan999/tokopedia-product-reviews)

**Dataset Details**:
- **Size**: Contains thousands of Indonesian product reviews
- **Language**: Bahasa Indonesia (with mixed formal/informal text, slang, emoticons)
- **Features**: Review text, ratings, product information
- **Task**: Multi-class sentiment classification (Positive/Negative/Neutral)

**License & Usage**: 
This dataset is available on Kaggle under their standard terms of use. Please refer to the [original dataset page](https://www.kaggle.com/datasets/farhan999/tokopedia-product-reviews) for specific licensing information.

**Citation**: If you use this dataset, please cite:
```
Farhan. (2019). Tokopedia Product Reviews. Kaggle. 
https://www.kaggle.com/datasets/farhan999/tokopedia-product-reviews
```

## Project Structures : 

```
.
├─ data/                                    # datasets
│  ├─ sample_reviews.csv                    # small sample for quick tests
│  └─ tokopedia-product-reviews-2019.csv
├─ model/                                   # saved artifacts
│  ├─ faiss.index                           # FAISS semantic search index
│  └─ lr_sentiment.pkl                      # trained Logistic Regression model
├─ sentiment.ipynb                          # main notebook (end-to-end pipeline)
├─ sentiment_pipeline_flowchart.png         # pipeline flowchart
├─ README.md                                # project documentation
├─ requirements.txt                         # project dependencies
└─ .gitignore                               # ignore venv, checkpoints, etc.
```

## Getting Started :

### Prerequisite
- Python 3.9+
- Internet Connection (for Hugging Face model download)

### Installation
1. Clone the Repository
```
git clone [https://github.com/your-username/tokopedia-sentiment-analysis.git](https://github.com/adikahnf/nolimit-ds-test-adika.git)
cd tokopedia-sentiment-analysis
```
2. Create a virtual environment
```
python -m venv venv
source venv/Scripts/Activate
```
3. Install dependencies
```
pip install -r requirements.txt
```

### Dataset setup 
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/farhan999/tokopedia-product-reviews)
2. Place the CSV file in the `data/` directory as `tokopedia-product-reviews-2019.csv`

### Running the Notebook

1. **Open and run**: `classification_tokopedia.ipynb`

## Model Details

### Primary Models Used:

1. **Classification Model**: 
   - Base: `Logistic Regression` and `Calibrated Linear SVC`
   - Input : SentenceTransformer embeddings (`float32`, normalized)
   - Task : 3-class classification (`Positive`/`Negative`/`Neutral`)

2. **Embedding Model**:
   - [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
   - Optimized for multilingual semantic similarity
   - Used both classifier features and FAISS indexing

3. **Search & Retrieval**:
   - FAISS (Facebook AI Similarity Search)
   - `IndexFlatIP` on normalized embeddings -> equivalent to cosine similarity
   - Fast approximate nearest neighbor search

### Model Results & Performance
| Metric (Test Set) | Logistic Regression | Calibrated Linear SVC |
|-------------------|---------------------|-----------------------|
| **Accuracy**      | ~0.76              | **~0.93** |
| **Macro F1**      | ~0.45              | **~0.54** |
| **Weighted F1**   | ~0.82              | **~0.83** |

* **Positive class**: Very high precision/recall (~0.95+) because it dominates the dataset.  
* **Neutral & Negative classes**: Lower precision/recall (0.15–0.35) due to limited samples.

The **macro F1** (simple average of F1 per class) is lower than the weighted F1 because it gives equal importance to the minority classes where performance is weaker.
**Weighted F1** is higher because it’s dominated by the majority positive class.

### Per-Class Breakdown (Test Set)

| Class     | Precision | Recall | F1-score |
|-----------|-----------|-------|---------|
| Positive  | ~0.97     | ~0.78 | ~0.87  |
| Neutral   | ~0.10     | ~0.40 | ~0.16  |
| Negative  | ~0.22     | ~0.68 | ~0.33  |

### Interpretation

- **Positive Class:**  
  The model performs very strongly on the **Positive** class, which is also the dominant class in the dataset. High precision (~0.97) indicates that when the model predicts positive, it is almost always correct, and the recall (~0.78) shows it captures most positive samples.

- **Neutral & Negative Classes:**  
  Performance is much lower for **Neutral** and **Negative** classes due to their **tiny sample sizes**. This data imbalance leads to difficulty in correctly identifying these minority classes, reflected in low precision and F1-scores. However, recall for Negative (~0.68) is relatively higher, indicating the model can find a fair portion of negative samples but with many false positives.

- **Model Improvements:**  
  Applying **class weighting** and using a stronger classifier (Calibrated Linear SVC) improved **minority recall** compared to the logistic baseline. The **macro-F1 score** captures the imbalance more accurately than accuracy, highlighting the gap between dominant and minority class performance.

### Class Imbalance
The Tokopedia dataset is **heavily skewed**:
* **Positive** reviews ≈ 90% of all samples  
* **Neutral** reviews ≈ 7%  
* **Negative** reviews ≈ 3%

Without adjustment, a model could predict “positive” for everything and still achieve high accuracy.  
To counter this:
* **Class Weights**: Applied `class_weight={"negative":6.0, "neutral":3.0, "positive":1.0}` in SVC to give minority classes more influence.
* **Downsampling Experiments**: Tried reducing positive samples to balance, but best results came from class weighting + full data.

### Model Justification:
- **SentenceTransformer embeddings**: Modern NLP approach that captures semantic meaning without heavy preprocessing (no stemming/stopwords required).
- **Calibrated Linear SVC**: Robust on imbalanced classes; calibration provides reliable class probabilities.
- **Logistic Regression (baseline)**: Simple, interpretable baseline to verify embedding quality.
- **FAISS**: Fast, scalable similarity search enabling retrieval-augmented predictions and transparent neighbor evidence.

## 📄 License
The **dataset is licensed under [MIT](https://opensource.org/license/mit)** 
# 🧠 Depression Text Classification using Machine Learning

This project focuses on classifying Reddit text posts into two categories: depression-related and non-depression-related. The goal is to build a robust machine learning pipeline to detect signs of depression from text using natural language processing and classification models.

Dataset used:  
🔗 [Depression Reddit Cleaned - Kaggle](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned/data)

---

## 📁 Dataset Overview

The dataset consists of cleaned Reddit posts labeled as:
- **1**: Indicates depression-related post  
- **0**: Indicates non-depression-related post  

---

## 🧹 Data Cleaning

- Duplicate entries were removed to avoid data leakage and model bias.
- Missing values were checked and handled accordingly (none found in the dataset).

---

## ✨ Text Preprocessing

To enhance model performance, the following text preprocessing steps were applied:

- **Case Folding**: Converted all characters to lowercase.
- **Punctuation Removal**: Removed punctuation to normalize the text.
- **Number Removal and Tokenization**: Removed numeric characters and split text into individual words.
- **Stopwords Removal**: Eliminated common words that do not carry significant meaning (e.g., “and”, “the”, etc.).
- **Stemming and Lemmatization**: Words were reduced to their root forms to standardize word variations.

These steps help to clean, normalize, and reduce noise in the text data, making it suitable for feature extraction.

---

## 🔢 Feature Extraction

Text was transformed using TF-IDF (Term Frequency-Inverse Document Frequency), which gives weight to important words while reducing the influence of commonly used terms.

---

## ⚖️ Handling Class Imbalance

Although the dataset is relatively balanced, SMOTE (Synthetic Minority Over-sampling Technique) was optionally used to ensure both classes are equally represented during training, helping to improve model fairness and accuracy.

---

## 🧠 Machine Learning Models

Three different classification models were trained and evaluated on the dataset:

| Model               | Train Accuracy | Test Accuracy |
|--------------------|----------------|---------------|
| **Support Vector Machine (SVM)** | 99.77%         | **96.08%**     |
| **Random Forest**                | 99.89%         | 95.62%         |
| **Decision Tree**               | 99.89%         | 92.61%         |

---

## 📊 Evaluation Summary

### Support Vector Machine (SVM)
- Achieved the highest test accuracy.
- Small gap between training and test accuracy indicates strong generalization.
- Very effective for high-dimensional and sparse data like text.

### Random Forest
- Performed consistently well.
- Slightly lower test accuracy than SVM but with high stability.
- Good at capturing complex patterns and reducing variance.

### Decision Tree
- High training accuracy but significantly lower test accuracy.
- Indicates possible overfitting.
- Useful when interpretability is a top priority.

---

## ✅ Conclusion

Among the three models tested, **SVM** showed the best performance in terms of generalization and accuracy on unseen data. **Random Forest** also proved to be a strong competitor with stable results. While **Decision Tree** is simpler and easier to interpret, it may require further tuning to reduce overfitting.

This text classification pipeline demonstrates the potential of machine learning in identifying mental health indicators from text, which can be further explored for real-world applications.

# Machine Learning Internship Tasks - Month 1

This repository contains solutions for Month 1 machine learning internship tasks:

## 📋 Tasks Completed

### Task 1: Email Spam Classification
- **Goal:** Classify emails as spam or ham (not spam)
- **Dataset:** SMS Spam Collection (5,574 messages)
- **Approach:**
  - Text preprocessing (lowercase, remove punctuation, stemming, stopwords)
  - TF-IDF vectorization for feature extraction
  - Multiple models compared (Naive Bayes, Logistic Regression, SVM, Random Forest)
  - Hyperparameter tuning for optimal performance
- **Best Model:** [Your best model name]
- **Performance:** 
  - Accuracy: ~98%
  - F1-Score: ~0.97
  - Precision: ~0.99
  - Recall: ~0.96

### Task 2: MNIST Digit Recognition
- **Goal:** Recognize handwritten digits (0-9)
- **Dataset:** MNIST (70,000 28x28 grayscale images)
- **Approach:**
  - Pixel normalization (0-255 → 0-1)
  - Multiple models compared (KNN, Random Forest, Neural Network)
  - Evaluation with confusion matrix and classification report
- **Best Model:** [Your best model name]
- **Performance:** 
  - Accuracy: ~97%

## 🛠️ Technologies Used
- Python 3.x
- Google Colab
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- NLTK for text processing

## 📁 Files in this Repository
- `Task1_Email_Spam_Classification.ipynb` - Complete code for Task 1
- `Task2_MNIST_Digit_Recognition.ipynb` - Complete code for Task 2
- `best_spam_classifier.pkl` - Saved spam classification model
- `mnist_best_model.pkl` - Saved MNIST model
- `spam_classifier_report.txt` - Performance report for Task 1

## 🚀 How to Run
1. Open the notebooks in Google Colab
2. Run all cells sequentially
3. For Task 1, all required datasets will be downloaded automatically
4. For Task 2, MNIST dataset loads via scikit-learn

## 📊 Results Summary
Both tasks achieved excellent performance with:
- High accuracy (>95%)
- Robust preprocessing pipelines
- Multiple model comparisons
- Comprehensive evaluation metrics

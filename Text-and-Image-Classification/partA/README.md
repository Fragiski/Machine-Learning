# Part A: Statistical Text Classification on IMDB

A binary sentiment classification pipeline evaluating statistical models on the IMDB dataset of 50,000 movie reviews.

## Overview

The goal is to classify movie reviews as positive or negative using traditional machine learning algorithms and sparse text representations. The pipeline compares Bernoulli Naive Bayes against regularized Logistic Regression.

## Implementation Details

- Dataset preprocessing: Reviews are loaded and split into training (80%) and development (20%) sets. Final assessment is performed on an unseen test set of 25,000 balanced reviews.
- Vocabulary filtering: Word frequencies are calculated to strip out the n most frequent stop words and k rarest tokens, eliminating noisy and non-informative vocabulary.
- Feature extraction: Text samples are transformed into binary document-term matrices using CountVectorizer(binary=True), recording word presence rather than raw frequency.
- Feature selection: Top m features with the highest discriminative power are selected via SelectKBest combined with mutual information scoring (mutual_info_classif).
- Model training: Evaluates Bernoulli Naive Bayes and Logistic Regression with L2 regularization control (parameter C).
- Learning curves: Evaluates Precision, Recall, and F1 across training subset fractions to monitor convergence and check for overfitting.

## Best Configuration and Results

The best experimental setup uses:
- Most frequent words pruned (n): 200
- Least frequent words pruned (k): 800
- Selected features (m): 4000
- Logistic Regression regularization (C): 0.1

Test Set Metrics:
- Accuracy: 0.8714 (87.14%)
- Negative class F1: 0.8708
- Positive class F1: 0.8720
- Macro Average F1: 0.8714

### Academic Context
Developed as a group project, coursework for the "Artificial Intelligence" course, Academic Year 2025-26.

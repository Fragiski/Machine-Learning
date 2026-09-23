# Part B: Deep Learning for Sentiment Analysis (Bi-GRU)

A recurrent neural network implementation in PyTorch for sentiment analysis on the IMDB movie review benchmark.

## Architecture

The network replaces bag-of-words representations with sequential modeling and pretrained semantic embeddings:
- Preprocessing: Vocabulary is restricted to the top 10,000 most frequent tokens. Reviews are padded or truncated to a fixed sequence length of 200 tokens using <PAD> and <UNK> markers .
- Embedding Layer: Initialized with 300-dimensional pretrained Word2Vec vectors (Google News 300), mapping tokens to dense semantic coordinates.
- Encoder: A 2-layer stacked Bidirectional Gated Recurrent Unit (Bi-GRU) that processes text forward and backward to capture directional dependencies .
- Pooling: Global Max Pooling extracts the most salient activation across the sequence hidden states .
- Regularization and Output: Dropout layer followed by a linear classification head optimized via Cross Entropy and Adam .

## Hyperparameter Configuration

- Hidden dimension: 128 units 
- Recurrent layers: 2 
- Batch size: 64 
- Optimizer: Adam 
- Sequence length: 200 

Training logs indicate rapid convergence within early epochs, followed by typical recurrent overfitting on training data as epoch count increases.

## Test Set Results

- Accuracy: 0.8600 (86%) 
- Negative class F1: 0.86 
- Positive class F1: 0.86 
- Macro F1: 0.8598 
- Micro F1: 0.8598 

### Academic Context
Developed as a group project, coursework for the "Artificial Intelligence" course, Academic Year 2025-26.

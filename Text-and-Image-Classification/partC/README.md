# Part C: Image Classification on FashionMNIST using CNN

A convolutional neural network built in PyTorch to classify 28x28 grayscale clothing images across 10 categories.

## Architecture

The system uses a CNN feature extractor coupled with a multi-layer perceptron (MLP) classifier:
- Input pipeline: 70,000 images split into 48,000 training, 12,000 development, and 10,000 test samples. Pixel values are normalized into the [-1, 1] range.
- Data Augmentation: Random horizontal flips and small random rotations (10 degrees) applied during training to improve generalization.
- Conv Block 1: 32 filters (3x3 kernel), ReLU activation, followed by MaxPool2d (downsampling from 28x28 to 14x14).
- Conv Block 2: 64 filters (3x3 kernel), ReLU activation, followed by MaxPool2d (downsampling from 14x14 to 7x7).
- Flattening: Produces a 3,136-dimensional feature vector (64 channels x 7 x 7).
- Dense Head: Linear layer mapping 3,136 inputs to 128 hidden units with ReLU and Dropout (0.3).
- Output Layer: 10 logits evaluated using CrossEntropyLoss.

## Performance and Results

The selected architecture (128 hidden units, 0.3 dropout) reached its optimal development loss at epoch 9.

Test Set Evaluation:
- Overall Accuracy: 0.9100 (91%)
- Macro F1: 0.9107
- Micro F1: 0.9112

Class breakdown highlights high recognition on geometrically distinct items (Trouser: 0.99 F1, Bag: 0.99 F1, Sandal: 0.98 F1), while top-wear categories with overlapping silhouettes present higher confusion (Shirt: 0.73 F1).

## Requirements and Execution

Requires Python 3, PyTorch, torchvision, numpy, and matplotlib.

Run the script:
python PartC.py

### Academic Context
Developed as a group project, coursework for the "Artificial Intelligence" course, Academic Year 2025-26.

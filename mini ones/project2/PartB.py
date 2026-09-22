import torch
import numpy as np
import os
from collections import Counter
from sklearn.model_selection import train_test_split
import gensim.downloader as api
import torch.nn as nn
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

#Loading data
def load_imdb_data(directory): 
    texts, labels = [], []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(directory, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    return texts, labels


all_train_texts, all_train_labels = load_imdb_data('C:/Users/apasa/Desktop/aclImdb/train')


#Splitting in training and development data
train_texts, dev_texts, train_labels, dev_labels = train_test_split(
    all_train_texts, all_train_labels, test_size=0.2, random_state=42)


#Making of the vocabulary
def build_vocab(texts, max_vocab_size=10000):

    all_words = " ".join(texts).lower().split()
    word_counts = Counter(all_words)
    
    #Keeps 10000 most often words
    common_words = word_counts.most_common(max_vocab_size)
    
    #Mapping
    vocab = {word: i+2 for i, (word, _) in enumerate(common_words)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab



#Converting reviews into sequences of numbers
def text_to_sequences(texts, vocab, max_len=200):
    sequences = []
    for text in texts:
        
        #If the word doesn't exist, 1 is added ("UNK")
        seq = [vocab.get(word.lower(), 1) for word in text.split()]
        
        if len(seq) < max_len:
            seq += [0] * (max_len - len(seq)) #Adding 0's at the end
        else:
            seq = seq[:max_len] #Cutting if the sequence is large
        sequences.append(seq)
    return torch.LongTensor(sequences)




#Loading word2vec through gensim
def load_word2vec_embeddings(vocab, embedding_dim=300):
    print("--- Downloading/Loading Word2Vec model (Google News 300)... ---")
    
    wv = api.load('word2vec-google-news-300')
    
    #Creating of the final weight board for PyTorch
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    
    for word, i in vocab.items():
        if word in wv:
            embedding_matrix[i] = wv[word]
        else:
            
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))#For words that do not exist
            
    return torch.FloatTensor(embedding_matrix)



#Preparation
print("--- STARTING THE PROCESS FOR PART B ---")

#Making of the vocabulary
vocab = build_vocab(train_texts)
print(f"Vocab size: {len(vocab)}")

#Converting texts into tensors
X_train_rnn = text_to_sequences(train_texts, vocab)
X_dev_rnn = text_to_sequences(dev_texts, vocab)
y_train_rnn = torch.FloatTensor(train_labels)
y_dev_rnn = torch.FloatTensor(dev_labels)

#Loadings embeddings
emb_matrix = load_word2vec_embeddings(vocab)

print("\n--- THE PROCESS FINISHED ---")
print(f"X_train_rnn shape: {X_train_rnn.shape}") 
print(f"Embedding Matrix shape: {emb_matrix.shape}") 


#Νeural Network
class StackedBidirectionalRNN(nn.Module):
    def __init__(self, emb_matrix, hidden_dim):
        super(StackedBidirectionalRNN, self).__init__()
        
        #Loading weights (embedding layer) 
        self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=False)
        
        #Stacked Bidirectional GRU (2 levels)
        self.rnn = nn.GRU(input_size=300, 
                          hidden_size=hidden_dim, 
                          num_layers=2,          
                          bidirectional=True,     
                          batch_first=True,
                          dropout=0.5)
        
        #Final layer for predicting
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid() #Compressing the final output into a value between 0 and 1


    def forward(self, x):
        
        #Converting word indices to vectors
        embedded = self.embedding(x) 

        #Passing embeddings through the GRU
        rnn_out, _ = self.rnn(embedded)
        
        #Global Max Pooling: extracting the most important feature of the entire sentence
        pooled_out, _ = torch.max(rnn_out, dim=1) #From [batch, seq_len, dim] to [batch,dim]
        
        out = self.fc(pooled_out)
        return self.sigmoid(out)


#Initialization of the model
model = StackedBidirectionalRNN(emb_matrix, hidden_dim=128)
print(model)

#Definition of parameters
batch_size = 64
learning_rate = 0.005
epochs = 1 
 
#Putting data in  batches
train_dataset = TensorDataset(X_train_rnn, y_train_rnn)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
 
#Loss function, 
criterion = nn.BCELoss() 

#Adam optimizer, Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
 
#List for losses for each epoque
train_losses = []
dev_losses = []
 
print(f"--- Beginning of training (Epochs: {epochs}) ---")
 
for epoch in range(epochs):
    #Training
    model.train()
    epoch_train_loss = 0

    for inputs, labels in train_loader:
        
        optimizer.zero_grad() #Reseting gradients from previous steps
       
        outputs = model(inputs).squeeze() #generating predictions
       
        loss = criterion(outputs, labels) #computing the loss between prediction-reality
   
        loss.backward() # calculating gradients (backpropagation)
     
        optimizer.step() #updating model weights based on gradients
        epoch_train_loss += loss.item()

    #Evaluation of the development set
    model.eval() # sets evaluation mode
    with torch.no_grad(): #Disabling gradient calculation
        dev_outputs = model(X_dev_rnn).squeeze()
        dev_loss = criterion(dev_outputs, y_dev_rnn)

    #Saving calculated losses for the learning curves
    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    dev_losses.append(dev_loss.item())
    print(f"Εpoque {epoch+1}: Train Loss = {avg_train_loss:.4f}, Dev Loss = {dev_loss.item():.4f}")
 
#Learing curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(dev_losses, label='Development Loss')
plt.title('Loss Curves - Part B')
plt.xlabel('Epoques')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()



#Final evaluation

#Loading test data
test_texts, test_labels = load_imdb_data('C:/Users/apasa/Desktop/aclImdb/test')
X_test_rnn = text_to_sequences(test_texts, vocab)
y_test_rnn = torch.FloatTensor(test_labels)


model.eval() #Evaluation mode
with torch.no_grad():
    #Predictions for the entire test set
    y_pred_probs = model(X_test_rnn).squeeze()
    #Converting data into probabilities (0 or 1)
    y_pred = (y_pred_probs > 0.5).int().numpy()
    y_true = y_test_rnn.numpy()

#Statistic table
print("\n" + "="*30)
print("Final results (Test set)")
print("="*30)
print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

#Micro/Macro F1 score
print(f"Micro F1-score: {f1_score(y_true, y_pred, average='micro'):.4f}")
print(f"Macro F1-score: {f1_score(y_true, y_pred, average='macro'):.4f}")
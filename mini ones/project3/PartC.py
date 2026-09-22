import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, f1_score

#Data augmentation and normalization (transforming training data to make the model better)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(), #Randomly flip of images for variety 
    transforms.RandomRotation(10),      #Rotating images by 10 degrees
    transforms.ToTensor(),              
    transforms.Normalize((0.5,), (0.5,)) #Normalizing pixels to [-1,1]
])

#Only normalizing the test set
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#Downloading FashionMNIST dataset
train_dataset_full = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)

#Splitting training and development data
train_size = int(0.8 * len(train_dataset_full))
dev_size = len(train_dataset_full) - train_size
train_dataset, dev_dataset = random_split(train_dataset_full, [train_size, dev_size])

#DataLoaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Train images: {len(train_dataset)}")
print(f"Dev images: {len(dev_dataset)}")
print(f"Test images: {len(test_dataset)}")


#Neural Network
class FashionCNN(nn.Module):
    def __init__(self, hidden_units=128, dropout_rate=0.5):
        super(FashionCNN, self).__init__()
        
        #CNN Encoder: Feature extraction from images
        self.encoder = nn.Sequential(
            #First level: Captures low-level patterns
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #Reducing dimensions to 14x14
            
            #Second level: Detecting more complex shapes
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  #dimensions: 7x7
        )
        
        #MLP Classifier: final classification
        #64 channels * 7 * 7 = 3136 inputs
        self.classifier = nn.Sequential(
            nn.Flatten(), #Converting 2D features maps into a 1D vector
            nn.Linear(64 * 7 * 7, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate), #prevents overfitting
            nn.Linear(hidden_units, 10) #10 output classes
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

#Initialization of the model
model = FashionCNN(hidden_units=128, dropout_rate=0.3)
print(model)

#Settings
epochs = 10
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

train_losses, dev_losses = [], []

print(f"--- BEGINNING OF TRAINING FOR PART C (Epochs: {epochs}) ---")

for epoch in range(epochs):
    # Training
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad() #Clearing previous gradients
        outputs = model(images)  #prediction
        loss = criterion(outputs, labels)
        loss.backward() #Backpropagation
        optimizer.step() #Updating weights
        running_loss += loss.item()
    
    #Evaluation with the development set
    model.eval()
    dev_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): #Disabling gradient calculation
        for images, labels in dev_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            dev_loss += loss.item()

            #Calculating accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    

    #Calculating statistics
    avg_train_loss = running_loss / len(train_loader)
    avg_dev_loss = dev_loss / len(dev_loader)
    accuracy = 100 * correct / total
    
    train_losses.append(avg_train_loss)
    dev_losses.append(avg_dev_loss)
    
    print(f"Epoque {epoch+1}: Train Loss = {avg_train_loss:.4f}, Dev Loss = {avg_dev_loss:.4f}, Dev Acc = {accuracy:.2f}%")

#Learning curves
plt.plot(train_losses, label='Train Loss')
plt.plot(dev_losses, label='Dev Loss')
plt.legend()
plt.title('Loss Curves - Part C')
plt.show()


 
# --- FINAL EVALUATION IN TEST SET (PART C) ---
print("\n" + "="*40)
print("FINAL EVALUATION - FashionMNIST")
print("="*40)
 
model.eval() # Model in evaluation mode
y_true = []
y_pred = []
 
with torch.no_grad(): #Disabling gradient calculation
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())
 
#10 categories of FashionMNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
 
#Statistics (Precision, Recall, F1)
print(classification_report(y_true, y_pred, target_names=class_names))
 
#Micro-Macro F1-scores
print(f"Micro F1-score: {f1_score(y_true, y_pred, average='micro'):.4f}")
print(f"Macro F1-score: {f1_score(y_true, y_pred, average='macro'):.4f}")
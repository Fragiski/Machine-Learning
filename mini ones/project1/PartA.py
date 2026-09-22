import sklearn
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


#Function created in order to manage large data
def load_imdb_data(directory): 
    texts = []     
    labels = []  
    for label_type in ['pos', 'neg']: 
        dir_name = os.path.join(directory, label_type)
        for fname in os.listdir(dir_name): 
            if fname.endswith('.txt'):     
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1 if label_type == 'pos' else 0)
    return texts, labels



my_path = 'C:/Users/apasa/Desktop/aclImdb/train' 

print("--- LOADING DATA ---")
all_texts, all_labels = load_imdb_data(my_path)

if len(all_texts) > 0:
    print(f"\nSUCCESSS!")
    print(f"Pos {len(all_texts)}")
    print(f"Neg {len(all_labels)}")
    print(f"First Review {all_texts[0][:100]}...")
else:
    print("\nSomething went wrong.")



#Loading the train data
all_train_texts, all_train_labels = load_imdb_data('C:/Users/apasa/Desktop/aclImdb/train')


#Splitting the data into two categories, Training and Development
train_texts, dev_texts, train_labels, dev_labels = train_test_split(
    all_train_texts, all_train_labels, test_size=0.2, random_state=42)

#Vocabulary creation

#Frequency of each word
vectorizer_init = CountVectorizer(binary=True) 
X_train_init = vectorizer_init.fit_transform(train_texts)

#Calculating in how many reviews each word is found
word_counts = np.asarray(X_train_init.sum(axis=0)).flatten()
words = vectorizer_init.get_feature_names_out()

#Putting words in order based on frequency
sorted_indices = np.argsort(word_counts)[::-1] #From least to most rare

# Choosing n,k
n, k = 300, 800
stop_words_list = list(words[sorted_indices[:n]]) + list(words[sorted_indices[-k:]])

#No stop-words
vectorizer = CountVectorizer(binary=True, stop_words=stop_words_list)
X_train_filtered = vectorizer.fit_transform(train_texts)



m = 3500

selector = SelectKBest(score_func=mutual_info_classif, k=m)
X_train_final = selector.fit_transform(X_train_filtered, train_labels)



#Development data transformation
X_dev_counts = vectorizer.transform(dev_texts)
X_dev_final = selector.transform(X_dev_counts)


#Training with BernoulliNB


#Creation of the model
clf = BernoulliNB()

#Training
clf.fit(X_train_final, train_labels)

#Prediction
predictions = clf.predict(X_dev_final)

#Calculating results
print("--- Results NAIVE BAYES (Development Set) ---")
print(f"Accuracy:  {accuracy_score(dev_labels, predictions):.4f}")
print(f"Precision: {precision_score(dev_labels, predictions):.4f}")
print(f"Recall:    {recall_score(dev_labels, predictions):.4f}")
print(f"F1 Score:  {f1_score(dev_labels, predictions):.4f}")


#Training with Logistic Regression

#Creation of the model
lr_clf = LogisticRegression(C=0.1 , max_iter=1000)

#Training
lr_clf.fit(X_train_final, train_labels)

#Prediction
lr_predictions = lr_clf.predict(X_dev_final)

#Calculating results
print("\n--- Results LOGISTIC REGRESSION (Development Set) ---")
print(f"Accuracy:  {accuracy_score(dev_labels, lr_predictions):.4f}")
print(f"Precision: {precision_score(dev_labels, lr_predictions):.4f}")
print(f"Recall:    {recall_score(dev_labels, lr_predictions):.4f}")
print(f"F1 Score:  {f1_score(dev_labels, lr_predictions):.4f}")


#Setting variables
percentages = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0] #Training percentages
train_results = {'precision': [], 'recall': [], 'f1': []}
dev_results = {'precision': [], 'recall': [], 'f1': []}
sample_sizes = []

print("--- ΚΑΜΠΥΛΕΣ ΜΑΘΗΣΗΣ ---")

for p in percentages:
    #Selection subset of training data
    num_samples = int(p * X_train_final.shape[0])
    sample_sizes.append(num_samples)
    
    X_subset = X_train_final[:num_samples]
    y_subset = np.array(train_labels)[:num_samples]
    
    #Training model with optimal variables
    model = LogisticRegression(C=0.1, max_iter=1000)
    model.fit(X_subset, y_subset)
    
    #Predictions
    y_train_pred = model.predict(X_subset)
    y_dev_pred = model.predict(X_dev_final)
    
    for data, y_true, y_pred, results in [
        ('Train', y_subset, y_train_pred, train_results),
        ('Dev', dev_labels, y_dev_pred, dev_results)
    ]:
        results['precision'].append(precision_score(y_true, y_pred, pos_label=1))
        results['recall'].append(recall_score(y_true, y_pred, pos_label=1))
        results['f1'].append(f1_score(y_true, y_pred, pos_label=1))

#Diagramms
metrics = ['precision', 'recall', 'f1']
plt.figure(figsize=(15, 5))

for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    plt.plot(sample_sizes, train_results[metric], 'o-', label=f'Training {metric.capitalize()}')
    plt.plot(sample_sizes, dev_results[metric], 's-', label=f'Development {metric.capitalize()}')
    plt.xlabel('Number of Training Examples')
    plt.ylabel(metric.capitalize())
    plt.title(f'Learning Curve: {metric.capitalize()}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()


print("\n-- LOADING DATA --")

test_texts, test_labels = load_imdb_data('C:/Users/apasa/Desktop/aclImdb/test')


print("\n--- FINAL EVALUATION IN TEST SET ---")

#Transforning results
X_test_filtered = vectorizer.transform(test_texts)
X_test_final = selector.transform(X_test_filtered)

#Prediction
final_predictions = model.predict(X_test_final)

#Printing final results
print(f"Final Accuracy:  {accuracy_score(test_labels, final_predictions):.4f}")
print(f"Final Precision: {precision_score(test_labels, final_predictions, pos_label=1):.4f}")
print(f"Final Recall:    {recall_score(test_labels, final_predictions, pos_label=1):.4f}")
print(f"Final F1 Score:  {f1_score(test_labels, final_predictions, pos_label=1):.4f}")



print("\n--- FINAL RESULT BOARD (TEST SET) ---")

#Predicting in Test Set
test_predictions = model.predict(X_test_final)

#Making the tables
report = classification_report(test_labels, test_predictions, target_names=['Negative', 'Positive'], digits=4)

print(report)

0

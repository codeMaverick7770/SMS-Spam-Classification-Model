# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
# The dataset is assumed to be in a tab-separated values (TSV) format with two columns: label and message
msg = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])

# Preprocess the data
# Convert the labels into binary format (spam: 1, ham: 0)
msg['label'] = msg['label'].map({'spam': 1, 'ham': 0})

# Split the data into training and testing sets
# Use 80% of the data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(msg['message'], msg['label'], test_size=0.2, random_state=42)

# Vectorize the text data
# Convert the raw text into numerical feature vectors
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train the model
# Use the Multinomial Naive Bayes algorithm for training the model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vect)

# Evaluate the model's performance
# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

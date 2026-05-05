#!/usr/bin/env python3

""" Implement Naive Bayes classifier for spam detection using scikit-learn library """


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score


""" Load Data """
def load_data():
    data = pd.read_csv("spam_sms.csv")

    # Standardize column names to lowercase to avoid mismatches
    data.columns = [col.lower() for col in data.columns]

    # Ensure correct columns (if dataset doesn't have correct ones)
    if 'label' not in data.columns:
        data = data.iloc[:, :2]                      #take first two columns
        data.columns = ['label', 'message']          #renaming columns

    # Encode labels
    # " ham " ==> 0 (not spam)
    # " spam " ==> 1 (spam)
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    return data


""" Split Data """
def split_data(data):
    X = data['message']                #Input features (text messages)
    y = data['label']                  #Output labels (spam/ham)

    #Split into 80% training and 20% testing
    return train_test_split(X, y, test_size=0.2, random_state=42)


""" Vectorization to convert text into numerical features """
def vectorize_text(X_train, X_test):
    vectorizer = CountVectorizer(stop_words='english')                #creates a bag of words model

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, vectorizer


""" Train Model """
def train_model(X_train_vec, y_train):
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model


""" Evaluate Model """
def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))


""" Predict New Message (takes new raw msg and converts into human readable label) """
def predict_message(model, vectorizer, message):
    msg_vec = vectorizer.transform([message])
    pred = model.predict(msg_vec)

    return "Spam" if pred[0] == 1 else "Ham"


""" Main """
def main():
    # Load
    data = load_data()

    # Split
    X_train, X_test, y_train, y_test = split_data(data)

    # Vectorize
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    # Train
    model = train_model(X_train_vec, y_train)

    # Evaluate
    evaluate_model(model, X_test_vec, y_test)

    # Test sample
    msg = "Congratulations! You have won a prize"
    print("\nSample Prediction:", predict_message(model, vectorizer, msg))


# Run
if __name__ == "__main__":
    main()
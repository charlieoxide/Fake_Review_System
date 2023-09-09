import tkinter as tk
from tkinter import filedialog, messagebox
import random
import numpy as np
import matplotlib.pyplot as plt
import threading
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# import dataset 
def import_dataset():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.csv")])
    if file_path:
        thread = threading.Thread(target=process_dataset, args=(file_path,))
        thread.start()

# Processing the dataset 

def process_dataset(file_path):
    try:
        with open(file_path, "r") as file:
            # Define the accuracy threshold
            accuracy_threshold = 0.8

            # Initialize lists for positive and negative reviews
            positive_reviews = []
            negative_reviews = []

            for line in file:
                review = line.strip()
                accuracy = random.uniform(0, 1)
                if accuracy >= accuracy_threshold:
                    positive_reviews.append(review)
                else:
                    negative_reviews.append(review)

            # Determine the number of reviews to select-from each category
            sample_size = min(len(positive_reviews), len(negative_reviews))

            # Select a balanced random sample of reviews
            random_positive_sample = random.sample(positive_reviews, sample_size)
            random_negative_sample = random.sample(negative_reviews, sample_size)
            balanced_dataset = random_positive_sample + random_negative_sample

            # Update the review text box with the balanced dataset
            review_text.delete("1.0", tk.END)
            review_text.insert(tk.END, "\n".join(balanced_dataset))

            # Perform classification
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(balanced_dataset)
            y = np.concatenate((np.ones(len(random_positive_sample)), np.zeros(len(random_negative_sample))))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Support Vector Machine (SVM) classifier
            svm_clf = SVC()
            svm_clf.fit(X_train, y_train)
            svm_accuracy = svm_clf.score(X_test, y_test)
            svm_predictions = svm_clf.predict(X_test)

            # Logistic Regression classifier
            logreg_clf = LogisticRegression()
            logreg_clf.fit(X_train, y_train)
            logreg_accuracy = logreg_clf.score(X_test, y_test)
            logreg_predictions = logreg_clf.predict(X_test)

            # Random Forest classifier
            rf_clf = RandomForestClassifier()
            rf_clf.fit(X_train, y_train)
            rf_accuracy = rf_clf.score(X_test, y_test)
            rf_predictions = rf_clf.predict(X_test)

            # Classification report
            svm_report = classification_report(y_test, svm_predictions, target_names=['Negative', 'Positive'])
            logreg_report = classification_report(y_test, logreg_predictions, target_names=['Negative', 'Positive'])
            rf_report = classification_report(y_test, rf_predictions, target_names=['Negative', 'Positive'])

        def analyze_reviews():
            reviews = review_text.get("1.0", tk.END).strip().split("\n")
            positive_keywords = ['good', 'great', 'excellent']
            negative_keywords = ['bad', 'poor', 'terrible']

            positive_reviews = []
            negative_reviews = []

            for review in reviews:
                if any(keyword in review.lower() for keyword in positive_keywords):
                    positive_reviews.append(review)
                elif any(keyword in review.lower() for keyword in negative_keywords):
                    negative_reviews.append(review)

            if positive_reviews:
                messagebox.showinfo("Positive Reviews", "\n".join(positive_reviews))
            else:
                messagebox.showinfo("Positive Reviews", "No positive reviews found.")

            if negative_reviews:
                messagebox.showinfo("Negative Reviews", "\n".join(negative_reviews))
            else:
                messagebox.showinfo("Negative Reviews", "No negative reviews found.")


            # Plot histogram
            plt.figure(figsize=(6, 4))
            labels = ['Positive', 'Negative']
            values = [len(positive_reviews), len(negative_reviews)]
            plt.bar(labels, values)
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.title('Review Sentiment Distribution')
            plt.show()

            # Plot pie chart
            plt.figure(figsize=(6, 6))
            plt.pie(values, labels=labels, autopct='%1.1f%%')
            plt.title('Review Sentiment Proportion')
            plt.show()
        # Display classification report in message boxes
        messagebox.showinfo("SVM Classification Report", svm_report)
        messagebox.showinfo("Logistic Regression Classification Report", logreg_report)
        messagebox.showinfo("Random Forest Classification Report", rf_report)

    except IOError as e:
        messagebox.showerror("Error", str(e))
        
def analyze_reviews():
    reviews = review_text.get("1.0", tk.END).strip().split("\n")
    positive_keywords = ['good', 'great', 'excellent']
    negative_keywords = ['bad', 'poor', 'terrible']

    positive_reviews = []
    negative_reviews = []

    for review in reviews:
        if any(keyword in review.lower() for keyword in positive_keywords):
            positive_reviews.append(review)
        elif any(keyword in review.lower() for keyword in negative_keywords):
            negative_reviews.append(review)

    if positive_reviews:
        messagebox.showinfo("Positive Reviews", "\n".join(positive_reviews))
    else:
        messagebox.showinfo("Positive Reviews", "No positive reviews found.")

    if negative_reviews:
        messagebox.showinfo("Negative Reviews", "\n".join(negative_reviews))
    else:
        messagebox.showinfo("Negative Reviews", "No negative reviews found.")


    # Plot histogram
    plt.figure(figsize=(6, 4))
    labels = ['Positive', 'Negative']
    values = [len(positive_reviews), len(negative_reviews)]
    plt.bar(labels, values)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Review Sentiment Distribution')
    plt.show()

    # Plot pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title('Review Sentiment Proportion')
    plt.show()

# Create the main window
window = tk.Tk()
window.title("Fake Product Review Monitoring System")

# Create the review entry text box
review_label = tk.Label(window, text="Enter product reviews (one review per line):")
review_label.pack()
review_text = tk.Text(window, height=10, width=50)
review_text.pack()

# Create the import button
import_button = tk.Button(window, text="Import Dataset", command=import_dataset)
import_button.pack()

# Create the analyze button
analyze_button = tk.Button(window, text="Analyze Reviews", command=analyze_reviews)
analyze_button.pack()

# Start the main GUI loop
window.mainloop()

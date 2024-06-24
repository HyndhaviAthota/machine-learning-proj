Ml project code:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import warnings

# Load the data
df = pd.read_csv("C:/Users/HP/Downloads/admission.csv")

# Data cleaning
df = df.dropna()
df = df.drop_duplicates()

# Prepare data and split into train and test sets
X = df[['AIEEE Rank', '12th Marks', '10th Marks']]
y = df['College']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

logistic_regression = LogisticRegression(max_iter=50000)
logistic_regression.fit(X_train, y_train)

svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

# Calculate accuracies
rf_accuracy = random_forest.score(X_test, y_test)
dt_accuracy = decision_tree.score(X_test, y_test)
lr_accuracy = logistic_regression.score(X_test, y_test)
svm_accuracy = svm_classifier.score(X_test, y_test)

# Print accuracies
print("Accuracy of Random Forest Classifier:", rf_accuracy)
print("Accuracy of Decision Tree Classifier:", dt_accuracy)
print("Accuracy of Logistic Regression Classifier:", lr_accuracy)
print("Accuracy of SVM Classifier:", svm_accuracy)

# Function to suggest college based on rank, 12th marks, and 10th marks
def suggest_college(rank, marks_12th, marks_10th):
    user_data = np.array([[rank, marks_12th, marks_10th]])
    predicted_college = random_forest.predict(user_data)[0]
    return predicted_college

# Take input from the user
try:
    user_rank = int(input("Enter your AIEEE rank: "))
    user_12th_marks = float(input("Enter your 12th marks: "))
    user_10th_marks = float(input("Enter your 10th marks: "))
except ValueError:
    print("Please enter valid numerical values for rank, 12th marks, and 10th marks.")
    exit()

# Suggest college based on rank, 12th marks, and 10th marks
suggested_college = suggest_college(user_rank, user_12th_marks, user_10th_marks)

# Print the suggested college
print(f"Based on your rank, 12th marks, and 10th marks, we suggest {suggested_college}.")
import tkinter as tk
from tkinter import messagebox
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("C:/Users/HP/Downloads/admission.csv")

# Data cleaning
df = df.dropna()
df = df.drop_duplicates()

# Prepare data
X = df[['AIEEE Rank', '12th Marks', '10th Marks']]
y = df['College']

# Train Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X, y)

# Function to suggest college based on rank, 12th marks, and 10th marks
def suggest_college(rank, marks_12th, marks_10th):
    user_data = np.array([[rank, marks_12th, marks_10th]])
    predicted_college = random_forest.predict(user_data)[0]
    return predicted_college

# Function to handle button click event
def suggest_college_gui():
    global rank_entry, marks_12th_entry, marks_10th_entry  # Declare entry widgets as global variables
    try:
        rank = int(rank_entry.get())
        marks_12th = float(marks_12th_entry.get())
        marks_10th = float(marks_10th_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values for rank, 12th marks, and 10th marks.")
        return
    
    suggested_college = suggest_college(rank, marks_12th, marks_10th)
    messagebox.showinfo("College Suggestion", f"We suggest {suggested_college} based on your input.", icon='info')

# Create main window
window = tk.Tk()
window.title("College Suggestion")
window.configure(bg="#ADD8E6")  # Set background color
window.geometry("800x400")  # Set window size

# Create labels and entries for user input with centered text
label_font = ("Arial", 14, "bold")
entry_font = ("Arial", 12)

# Create labels and entries with centered text
labels = ["Enter your AIEEE rank:", "Enter your 12th marks:", "Enter your 10th marks:"]
entries = [tk.Entry(window, font=entry_font) for _ in range(3)]

for i, (label_text, entry) in enumerate(zip(labels, entries)):
    tk.Label(window, text=label_text, font=label_font, bg="#ADD8E6", anchor="center").grid(row=i, column=0, padx=10, pady=5)
    entry.grid(row=i, column=1, padx=10, pady=5)

# Assign entry widgets to global variables
rank_entry, marks_12th_entry, marks_10th_entry = entries

# Create button to suggest college with colorful styling
button_font = ("Arial", 14, "bold")
suggest_button = tk.Button(window, text="Suggest College", font=button_font, bg="#FF6347", fg="#FFFFFF", command=suggest_college_gui)
suggest_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="WE")

# Run the GUI
window.mainloop()

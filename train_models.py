import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pickle

def train_and_compare_models(csv_file='isl_landmarks_processed.csv'):
    print("Loading preprocessed dataset...")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Run 'preprocess.py' first!")
        return

    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine (SVM)": SVC(kernel='rbf'),
        "Naive Bayes": GaussianNB()
    }

    best_model_name = ""
    best_model = None
    highest_accuracy = 0.0
    results = []

    print("\n--- Training and Evaluating 5 Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results.append({"Model": name, "Accuracy (%)": round(acc * 100, 2)})
        print(f"-> {name} Accuracy: {acc * 100:.2f}%\n")
        
        if acc > highest_accuracy:
            highest_accuracy = acc
            best_model_name = name
            best_model = model

    results_df = pd.DataFrame(results)
    print("="*40)
    print(" MODEL PERFORMANCE COMPARISON SUMMARY ")
    print("="*40)
    print(results_df.to_string(index=False))
    print("="*40)
    
    print(f"\n🏆 Best Model: {best_model_name} with {highest_accuracy * 100:.2f}% accuracy!")

    with open('isl_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Saved best model to 'isl_model.pkl'.")

if __name__ == "__main__":
    train_and_compare_models()
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def train_random_forest():
    # Load train and test datasets
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    # Separate features and label
    X_train = train_df.drop(columns=["style"])
    y_train = train_df["style"]
    
    X_test = test_df.drop(columns=["style"])
    y_test = test_df["style"]
    
    # Encode the labels since they are strings
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Scale features (fit new scaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compute class weights for balanced performance
    classes = np.unique(y_train_encoded)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_encoded)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # Initialize Random Forest model
    rf = RandomForestClassifier(
        n_jobs=-1,
        random_state=42
    )
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [class_weight_dict]
    }
    
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_encoded)
    
    # Best model after grid search
    best_rf = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Cross-validation score
    cv_scores = cross_val_score(best_rf, X_train_scaled, y_train_encoded, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average Cross-validation score: {np.mean(cv_scores):.4f}")
    
    # Train the best model
    best_rf.fit(X_train_scaled, y_train_encoded)
    
    # Predict on train and test data
    y_train_pred = best_rf.predict(X_train_scaled)
    y_test_pred = best_rf.predict(X_test_scaled)
    
    # Evaluate train metrics
    print("Train Metrics:")
    print(f"Accuracy: {accuracy_score(y_train_encoded, y_train_pred):.4f}")
    print(f"Precision: {precision_score(y_train_encoded, y_train_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_train_encoded, y_train_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_train_encoded, y_train_pred, average='weighted'):.4f}")
    print("\n")
    
    # Evaluate test metrics
    print("Test Metrics:")
    print(f"Accuracy: {accuracy_score(y_test_encoded, y_test_pred):.4f}")
    print(f"Precision: {precision_score(y_test_encoded, y_test_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test_encoded, y_test_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_test_encoded, y_test_pred, average='weighted'):.4f}")
    
    # Confusion matrix for test set
    cm = confusion_matrix(y_test_encoded, y_test_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Save confusion matrix as PNG
    plt.savefig("output/conf_rf.png", bbox_inches="tight")
    print("Confusion matrix saved as 'output/conf_rf.png'")
    
    # Save the model, scaler, and label encoder for later use
    joblib.dump(best_rf, "models/rf_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    print("Model, scaler, and label encoder saved.")

if __name__ == "__main__":
    train_random_forest()
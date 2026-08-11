import os
import json
import joblib
import numpy as np
import pandas as pd
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.features import extract_features
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_data(data_dir):
    features = []
    labels = []
    filenames_list = []
    
    # Define mapping
    label_map = {
        'not_bullying': 0,
        'bullying': 1
    }
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.wav'):
            continue
            
        file_path = os.path.join(data_dir, filename)
        
        # Extract label from filename
        category = None
        for k in label_map.keys():
            if filename.startswith(k):
                category = k
                break
                
        if category:
            feat = extract_features(file_path)
            if feat is not None:
                features.append(feat)
                labels.append(label_map[category])
                filenames_list.append(filename)
                
    return np.array(features), np.array(labels), label_map, filenames_list

def main():
    print("Loading Training data...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(project_root, 'data', 'chunks', 'train')
    test_dir = os.path.join(project_root, 'data', 'chunks', 'test')
    
    X_train, y_train, label_map, train_filenames = load_data(train_dir)
    
    if len(X_train) == 0:
        print("No training data found. Please run scripts/chunk_audio.py first.")
        return
        
    print(f"Loaded {len(X_train)} training samples.")
    
    print("Loading Test data...")
    X_test, y_test, _, test_filenames = load_data(test_dir)
    
    if len(X_test) == 0:
        print("No test data found. Please ensure data/chunks/test has files.")
        return
        
    print(f"Loaded {len(X_test)} test samples.")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectFromModel
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
        'SVM (Raw)': SVC(kernel='rbf', C=100, gamma='scale', random_state=42),
        'SVM + PCA': Pipeline([
            ('pca', PCA(n_components=50)),
            ('svm', SVC(kernel='rbf', random_state=42))
        ]),
        'SVM + LDA': Pipeline([
            ('lda', LDA(n_components=1)), # LDA max components is n_classes - 1
            ('svm', SVC(kernel='rbf', random_state=42))
        ]),
        'SVM + Mutual Info Selection (Primary)': Pipeline([
            ('selector', SelectKBest(mutual_info_classif, k=50)),
            ('svm', SVC(kernel='rbf', random_state=42))
        ]),
        'SVM + RF Multivariate Selection': Pipeline([
            ('selector', SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), max_features=50)),
            ('svm', SVC(kernel='rbf', random_state=42))
        ])
    }
    
    metrics = {}
    best_model = None
    
    print("Training models...")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        metrics[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Confusion_Matrix': cm
        }
        
        print(f"{name} -> F1: {f1:.4f} | Acc: {acc:.4f}")
        
        if name == 'SVM + Mutual Info Selection (Primary)':
            best_model = model
            
    # Save metrics
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
        
    metrics_path = os.path.join(models_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")
    
    # Save best pipeline
    pipeline = {
        'scaler': scaler,
        'model': best_model,
        'label_map': label_map
    }
    joblib_path = os.path.join(models_dir, 'svm_model.joblib')
    joblib.dump(pipeline, joblib_path)
    print(f"Saved SVM pipeline to {joblib_path}")
    
    print("\n--- Error Analysis (Primary SVM Model) ---")
    y_pred_best = best_model.predict(X_test_scaled)
    reverse_map = {v: k for k, v in label_map.items()}
    misclassified = 0
    for i in range(len(y_test)):
        if y_test[i] != y_pred_best[i]:
            misclassified += 1
            true_label = reverse_map[y_test[i]]
            pred_label = reverse_map[y_pred_best[i]]
            print(f"❌ MISCLASSIFIED: {test_filenames[i]}")
            print(f"   True: {true_label} | Predicted: {pred_label}")
            
    if misclassified == 0:
        print("✅ No chunks were misclassified by the primary model!")

if __name__ == "__main__":
    main()

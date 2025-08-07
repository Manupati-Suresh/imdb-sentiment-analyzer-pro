import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
import glob
import time
import logging
from datetime import datetime
import json
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_imdb_data(data_dir):
    """Load IMDb movie reviews from pos/neg directories with progress tracking"""
    texts = []
    labels = []
    
    logger.info(f"Loading data from {data_dir}")
    
    # Load positive reviews
    pos_files = glob.glob(os.path.join(data_dir, 'pos', '*.txt'))
    logger.info(f"Found {len(pos_files)} positive review files")
    
    for i, file_path in enumerate(pos_files):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if content:  # Only add non-empty reviews
                    texts.append(content)
                    labels.append(1)  # positive
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
        
        if (i + 1) % 1000 == 0:
            logger.info(f"Loaded {i + 1} positive reviews")
    
    # Load negative reviews
    neg_files = glob.glob(os.path.join(data_dir, 'neg', '*.txt'))
    logger.info(f"Found {len(neg_files)} negative review files")
    
    for i, file_path in enumerate(neg_files):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
                if content:  # Only add non-empty reviews
                    texts.append(content)
                    labels.append(0)  # negative
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
        
        if (i + 1) % 1000 == 0:
            logger.info(f"Loaded {i + 1} negative reviews")
    
    logger.info(f"Total loaded: {len(texts)} reviews")
    return texts, labels

def create_visualizations(y_val, val_predictions, val_probabilities, model_name):
    """Create and save model performance visualizations"""
    
    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Model Performance Analysis - {model_name}', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_val, val_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_val, val_probabilities[:, 1])
    auc_score = roc_auc_score(y_val, val_probabilities[:, 1])
    axes[0,1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0,1].set_xlim([0.0, 1.0])
    axes[0,1].set_ylim([0.0, 1.05])
    axes[0,1].set_xlabel('False Positive Rate')
    axes[0,1].set_ylabel('True Positive Rate')
    axes[0,1].set_title('ROC Curve')
    axes[0,1].legend(loc="lower right")
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_val, val_probabilities[:, 1])
    axes[1,0].plot(recall, precision, color='blue', lw=2)
    axes[1,0].set_xlabel('Recall')
    axes[1,0].set_ylabel('Precision')
    axes[1,0].set_title('Precision-Recall Curve')
    axes[1,0].grid(True)
    
    # 4. Prediction Confidence Distribution
    positive_confidences = val_probabilities[y_val == 1, 1]
    negative_confidences = val_probabilities[y_val == 0, 0]
    
    axes[1,1].hist(positive_confidences, bins=30, alpha=0.7, label='Positive', color='green')
    axes[1,1].hist(negative_confidences, bins=30, alpha=0.7, label='Negative', color='red')
    axes[1,1].set_xlabel('Confidence Score')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Confidence Score Distribution')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'{model_name.lower()}_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance plots saved to {plots_dir}")

def evaluate_model(model, vectorizer, X_val, y_val, model_name):
    """Comprehensive model evaluation"""
    
    logger.info(f"Evaluating {model_name}...")
    
    # Transform validation data
    X_val_vec = vectorizer.transform(X_val)
    
    # Make predictions
    val_predictions = model.predict(X_val_vec)
    val_probabilities = model.predict_proba(X_val_vec)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, val_predictions)
    auc_score = roc_auc_score(y_val, val_probabilities[:, 1])
    
    # Classification report
    report = classification_report(y_val, val_predictions, output_dict=True)
    
    # Create visualizations
    create_visualizations(y_val, val_predictions, val_probabilities, model_name)
    
    # Log results
    logger.info(f"{model_name} Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  AUC Score: {auc_score:.4f}")
    logger.info(f"  Precision (Positive): {report['1']['precision']:.4f}")
    logger.info(f"  Recall (Positive): {report['1']['recall']:.4f}")
    logger.info(f"  F1-Score (Positive): {report['1']['f1-score']:.4f}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'classification_report': report,
        'predictions': val_predictions,
        'probabilities': val_probabilities
    }

def train_multiple_models(X_train, y_train, X_val, y_val, vectorizer):
    """Train and compare multiple models"""
    
    # Transform training data
    X_train_vec = vectorizer.fit_transform(X_train)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        start_time = time.time()
        
        # Train model
        model.fit(X_train_vec, y_train)
        
        training_time = time.time() - start_time
        logger.info(f"{name} training completed in {training_time:.2f} seconds")
        
        # Evaluate model
        result = evaluate_model(model, vectorizer, X_val, y_val, name)
        result['training_time'] = training_time
        result['model'] = model
        
        results[name] = result
    
    return results

def hyperparameter_tuning(X_train, y_train, vectorizer):
    """Perform hyperparameter tuning for the best model"""
    
    logger.info("Starting hyperparameter tuning...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # Parameter grid
    param_grid = {
        'tfidf__max_features': [5000, 10000, 15000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__min_df': [1, 2, 3],
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__max_iter': [1000]
    }
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def save_model_metadata(results, best_model_name, training_config):
    """Save model metadata and training information"""
    
    metadata = {
        'training_date': datetime.now().isoformat(),
        'best_model': best_model_name,
        'training_config': training_config,
        'model_performance': {}
    }
    
    # Add performance metrics for all models
    for name, result in results.items():
        metadata['model_performance'][name] = {
            'accuracy': result['accuracy'],
            'auc_score': result['auc_score'],
            'training_time': result['training_time'],
            'precision': result['classification_report']['1']['precision'],
            'recall': result['classification_report']['1']['recall'],
            'f1_score': result['classification_report']['1']['f1-score']
        }
    
    # Save metadata
    with open('model/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Model metadata saved to model/model_metadata.json")

def main():
    parser = argparse.ArgumentParser(description='Train IMDb sentiment analysis models')
    parser.add_argument('--data-dir', default='train', help='Directory containing training data')
    parser.add_argument('--compare-models', action='store_true', help='Compare multiple models')
    parser.add_argument('--tune-hyperparams', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--test-size', type=float, default=0.2, help='Validation set size')
    
    args = parser.parse_args()
    
    # Training configuration
    training_config = {
        'data_directory': args.data_dir,
        'test_size': args.test_size,
        'vectorizer_params': {
            'max_features': 10000,
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'min_df': 2
        },
        'compare_models': args.compare_models,
        'hyperparameter_tuning': args.tune_hyperparams
    }
    
    logger.info("Starting model training pipeline...")
    logger.info(f"Configuration: {training_config}")
    
    # Load data
    try:
        train_texts, train_labels = load_imdb_data(args.data_dir)
    except FileNotFoundError:
        logger.error(f"Training data directory '{args.data_dir}' not found!")
        logger.info("Please ensure you have the IMDb dataset in the specified directory.")
        return
    
    logger.info(f"Loaded {len(train_texts)} training samples")
    logger.info(f"Positive samples: {sum(train_labels)}")
    logger.info(f"Negative samples: {len(train_labels) - sum(train_labels)}")
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_texts, train_labels, 
        test_size=args.test_size, 
        random_state=42, 
        stratify=train_labels
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Create vectorizer
    logger.info("Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(**training_config['vectorizer_params'])
    
    if args.compare_models:
        # Train and compare multiple models
        results = train_multiple_models(X_train, y_train, X_val, y_val, vectorizer)
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_model = results[best_model_name]['model']
        
        logger.info(f"Best model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
        
    else:
        # Train single Logistic Regression model
        logger.info("Training Logistic Regression model...")
        X_train_vec = vectorizer.fit_transform(X_train)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_vec, y_train)
        
        # Evaluate
        result = evaluate_model(model, vectorizer, X_val, y_val, 'Logistic Regression')
        results = {'Logistic Regression': result}
        best_model_name = 'Logistic Regression'
        best_model = model
    
    # Hyperparameter tuning (optional)
    if args.tune_hyperparams:
        logger.info("Performing hyperparameter tuning...")
        tuned_model = hyperparameter_tuning(X_train, y_train, vectorizer)
        
        # Evaluate tuned model
        tuned_result = evaluate_model(
            tuned_model.named_steps['classifier'], 
            tuned_model.named_steps['tfidf'], 
            X_val, y_val, 
            'Tuned Logistic Regression'
        )
        
        results['Tuned Logistic Regression'] = tuned_result
        
        # Update best model if tuned version is better
        if tuned_result['accuracy'] > results[best_model_name]['accuracy']:
            best_model_name = 'Tuned Logistic Regression'
            best_model = tuned_model.named_steps['classifier']
            vectorizer = tuned_model.named_steps['tfidf']
    
    # Save best model and vectorizer
    logger.info(f"Saving best model: {best_model_name}")
    joblib.dump(best_model, 'model/imdb_model.pkl')
    joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
    
    # Save model metadata
    save_model_metadata(results, best_model_name, training_config)
    
    # Print final summary
    logger.info("=" * 50)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 50)
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Validation Accuracy: {results[best_model_name]['accuracy']:.4f}")
    logger.info(f"AUC Score: {results[best_model_name]['auc_score']:.4f}")
    logger.info("Model files saved to model/ directory")
    logger.info("You can now run the Streamlit app with: streamlit run app.py")

if __name__ == "__main__":
    main()
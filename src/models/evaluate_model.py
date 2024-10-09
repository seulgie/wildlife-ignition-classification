from sklearn.metrics import accuracy_score, classification_report

def evaluate_models(best_models, X_test, y_test):
    """
    Evaluate the trained models on the test set and return evaluation metrics.

    Args:
        best_models (dict): Best models found during training.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target variable.

    Returns:
        dict: Evaluation metrics for each model.
    """
    evaluation_results = {}
    
    # Loop through each best model and evaluate it
    for name, model in best_models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store evaluation results
        evaluation_results[name] = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1-score': report['weighted avg']['f1-score']
        }
        
        print(f"Evaluation metrics for {name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {report['weighted avg']['precision']:.4f}")
        print(f"Recall: {report['weighted avg']['recall']:.4f}")
        print(f"F1-Score: {report['weighted avg']['f1-score']:.4f}\n")
        
    return evaluation_results

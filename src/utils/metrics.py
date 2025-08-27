import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, accuracy_score
)
from sklearn.calibration import calibration_curve
import torch


def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate comprehensive metrics for binary classification."""
    
    # Convert probabilities to predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Clinical metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    # Additional metrics
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value
    ppv = precision  # Positive predictive value (same as precision)
    
    # Average precision (area under PR curve)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'npv': npv,
        'ppv': ppv,
        'avg_precision': avg_precision,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def find_optimal_threshold(y_true, y_pred_proba, target_sensitivity=0.95):
    """Find optimal threshold to achieve target sensitivity."""
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Find threshold that achieves target sensitivity
    valid_indices = tpr >= target_sensitivity
    if not np.any(valid_indices):
        # If target sensitivity not achievable, return threshold for max sensitivity
        best_idx = np.argmax(tpr)
        return thresholds[best_idx], tpr[best_idx], 1 - fpr[best_idx]
    
    # Among valid thresholds, choose one with highest specificity
    valid_specificities = 1 - fpr[valid_indices]
    valid_thresholds = thresholds[valid_indices]
    valid_sensitivities = tpr[valid_indices]
    
    best_idx = np.argmax(valid_specificities)
    
    return (
        valid_thresholds[best_idx],
        valid_sensitivities[best_idx],
        valid_specificities[best_idx]
    )


def plot_roc_curve(y_true, y_pred_proba, save_path=None, title="ROC Curve"):
    """Plot ROC curve with AUC score."""
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None, title="Precision-Recall Curve"):
    """Plot precision-recall curve with average precision score."""
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AP = {avg_precision:.3f})')
    
    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='red', linestyle='--', 
                label=f'Random (AP = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_confusion_matrix(y_true, y_pred, class_names=['Uninfected', 'Parasitized'], 
                         save_path=None, title="Confusion Matrix"):
    """Plot confusion matrix with percentages."""
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    # Add raw counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'({cm[i, j]})', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_calibration_curve(y_true, y_pred_proba, n_bins=10, save_path=None, 
                          title="Calibration Curve"):
    """Plot calibration curve to assess probability calibration."""
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
             label="Model", color='blue')
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def clinical_evaluation_report(y_true, y_pred_proba, target_sensitivity=0.95):
    """Generate comprehensive clinical evaluation report."""
    
    # Find optimal threshold
    optimal_threshold, achieved_sensitivity, achieved_specificity = find_optimal_threshold(
        y_true, y_pred_proba, target_sensitivity
    )
    
    # Calculate metrics at optimal threshold
    metrics_optimal = calculate_metrics(y_true, y_pred_proba, optimal_threshold)
    
    # Calculate metrics at default threshold (0.5)
    metrics_default = calculate_metrics(y_true, y_pred_proba, 0.5)
    
    report = {
        'optimal_threshold': optimal_threshold,
        'target_sensitivity': target_sensitivity,
        'achieved_sensitivity': achieved_sensitivity,
        'achieved_specificity': achieved_specificity,
        'metrics_at_optimal_threshold': metrics_optimal,
        'metrics_at_default_threshold': metrics_default,
        'auc': roc_auc_score(y_true, y_pred_proba),
        'average_precision': average_precision_score(y_true, y_pred_proba)
    }
    
    return report


def print_clinical_report(report):
    """Print formatted clinical evaluation report."""
    
    print("=" * 60)
    print("CLINICAL EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\nTarget Sensitivity: {report['target_sensitivity']:.1%}")
    print(f"Optimal Threshold: {report['optimal_threshold']:.4f}")
    print(f"Achieved Sensitivity: {report['achieved_sensitivity']:.1%}")
    print(f"Achieved Specificity: {report['achieved_specificity']:.1%}")
    
    print(f"\nOverall Performance:")
    print(f"  AUC: {report['auc']:.4f}")
    print(f"  Average Precision: {report['average_precision']:.4f}")
    
    print(f"\nMetrics at Optimal Threshold ({report['optimal_threshold']:.4f}):")
    opt_metrics = report['metrics_at_optimal_threshold']
    print(f"  Accuracy: {opt_metrics['accuracy']:.1%}")
    print(f"  Sensitivity: {opt_metrics['sensitivity']:.1%}")
    print(f"  Specificity: {opt_metrics['specificity']:.1%}")
    print(f"  Precision (PPV): {opt_metrics['precision']:.1%}")
    print(f"  NPV: {opt_metrics['npv']:.1%}")
    print(f"  F1-Score: {opt_metrics['f1']:.4f}")
    
    print(f"\nConfusion Matrix (Optimal Threshold):")
    print(f"  True Negatives: {opt_metrics['tn']}")
    print(f"  False Positives: {opt_metrics['fp']}")
    print(f"  False Negatives: {opt_metrics['fn']}")
    print(f"  True Positives: {opt_metrics['tp']}")
    
    print(f"\nClinical Interpretation:")
    if opt_metrics['sensitivity'] >= report['target_sensitivity']:
        print(f"  ✓ Sensitivity target achieved ({opt_metrics['sensitivity']:.1%} >= {report['target_sensitivity']:.1%})")
    else:
        print(f"  ✗ Sensitivity target not met ({opt_metrics['sensitivity']:.1%} < {report['target_sensitivity']:.1%})")
    
    if opt_metrics['specificity'] >= 0.85:
        print(f"  ✓ Good specificity ({opt_metrics['specificity']:.1%})")
    else:
        print(f"  ⚠ Low specificity ({opt_metrics['specificity']:.1%}) - may cause over-treatment")
    
    print("=" * 60)


class MetricsTracker:
    """Track metrics during training and validation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, predictions, targets):
        """Update with batch predictions and targets."""
        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())
    
    def compute(self, threshold=0.5):
        """Compute metrics for all accumulated predictions."""
        if len(self.predictions) == 0:
            return {}
        
        y_true = np.array(self.targets)
        y_pred_proba = np.array(self.predictions)
        
        return calculate_metrics(y_true, y_pred_proba, threshold)
    
    def compute_clinical_report(self, target_sensitivity=0.95):
        """Compute clinical evaluation report."""
        if len(self.predictions) == 0:
            return {}
        
        y_true = np.array(self.targets)
        y_pred_proba = np.array(self.predictions)
        
        return clinical_evaluation_report(y_true, y_pred_proba, target_sensitivity)


if __name__ == "__main__":
    # Test metrics with dummy data
    np.random.seed(42)
    
    # Generate dummy predictions
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive class
    y_pred_proba = np.random.beta(2, 5, n_samples)  # Skewed probabilities
    
    # Make predictions somewhat correlated with true labels
    y_pred_proba = 0.7 * y_true + 0.3 * y_pred_proba
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    
    print("Testing metrics calculation...")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred_proba)
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    
    # Clinical report
    report = clinical_evaluation_report(y_true, y_pred_proba)
    print_clinical_report(report)

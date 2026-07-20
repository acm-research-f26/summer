# utils/metrics.py
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, confusion_matrix
import numpy as np

def calculate_agreement_metrics(y_true, y_pred, stage_labels=None):
    """
    Calculates Cohen's Kappa and F1 scores for sleep staging agreement.
    """
    if stage_labels is None:
        stage_labels = list(set(y_true) | set(y_pred))

    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=stage_labels, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=stage_labels)

    return {
        'cohen_kappa': kappa,
        'f1_score': f1,
        'accuracy': acc,
        'confusion_matrix': conf_matrix
    }
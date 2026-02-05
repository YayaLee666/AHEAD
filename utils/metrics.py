import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, ndcg_score,
    precision_score, recall_score, matthews_corrcoef, accuracy_score,
    balanced_accuracy_score
)
from sklearn.metrics import fbeta_score
import numpy as np



def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'average_precision': ..., 'roc_auc': ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    # Average Precision 
    average_precision = average_precision_score(y_true=labels, y_score=predicts)

    try:
        roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
    except ValueError as e:
        if 'Only one class present' in str(e):
            roc_auc = float('nan') 
        else:
            raise e

    return {'average_precision': average_precision, 'roc_auc': roc_auc}



from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, fbeta_score,
    precision_score, matthews_corrcoef, balanced_accuracy_score,
    ndcg_score, precision_recall_curve
)
import numpy as np
import torch

def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    metrics = {}

    # Ranking-based metrics (do not depend on threshold)
    metrics["AUROC"] = roc_auc_score(labels, predicts)
    metrics["AUPRC"] = average_precision_score(labels, predicts)

    # Compute best threshold based on F1
    precision, recall, thresholds = precision_recall_curve(labels, predicts)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    metrics["best_threshold"] = best_threshold
    metrics["best_f1"] = best_f1

    # Apply best threshold
    predicts_label = (predicts >= best_threshold).astype(int)

    # Classification metrics
    metrics["f1"] = f1_score(labels, predicts_label)
    metrics["f0.5"] = fbeta_score(labels, predicts_label, beta=0.5)
    metrics["f2"] = fbeta_score(labels, predicts_label, beta=2.0)
    metrics["precision"] = precision_score(labels, predicts_label, zero_division=0)
    metrics["mcc"] = matthews_corrcoef(labels, predicts_label)
    metrics["balanced_accuracy"] = balanced_accuracy_score(labels, predicts_label)

    # Recall@K
    k = int(np.sum(labels))
    if k > 0:
        top_k_indices = predicts.argsort()[-k:]
        metrics["recall@k"] = np.sum(labels[top_k_indices]) / k
    else:
        metrics["recall@k"] = 0.0

    # NDCG
    try:
        metrics["ndcg"] = ndcg_score([labels], [predicts])
    except:
        metrics["ndcg"] = float("nan")

    return metrics


def get_edge_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    metrics = {}
    metrics["AUROC"] = roc_auc_score(labels, predicts)
    metrics["AUPRC"] = average_precision_score(labels, predicts)

    # Compute best threshold based on F1
    precision, recall, thresholds = precision_recall_curve(labels, predicts)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    metrics["best_threshold"] = best_threshold
    metrics["best_f1"] = best_f1

    # Apply best threshold
    predicts_label = (predicts >= best_threshold).astype(int)

    metrics["f1"] = f1_score(labels, predicts_label)
    metrics["f0.5"] = fbeta_score(labels, predicts_label, beta=0.5)
    metrics["f2"] = fbeta_score(labels, predicts_label, beta=2.0)
    metrics["precision"] = precision_score(labels, predicts_label)
    # metrics["recall"] = recall_score(labels, predicts_label)
    metrics["mcc"] = matthews_corrcoef(labels, predicts_label)
    metrics["balanced_accuracy"] = balanced_accuracy_score(labels, predicts_label)

    # Ranking metrics
    k = int(np.sum(labels))
    if k > 0:
        top_k_indices = predicts.argsort()[-k:]
        metrics["recall@k"] = np.sum(labels[top_k_indices]) / k
    else:
        metrics["recall@k"] = 0.0

    try:
        metrics["ndcg"] = ndcg_score([labels], [predicts])
    except:
        metrics["ndcg"] = float("nan")

    return metrics
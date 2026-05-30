import numpy as np

def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def classification_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    precisions = []
    recalls = []
    f1_scores = []
    
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1_score": np.mean(f1_scores)
    }


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2) 
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) 
    
    if ss_tot == 0:
        return 0.0
        
    return 1 - (ss_res / ss_tot)

def adjusted_r2_score(y_true, y_pred, num_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = num_features
    
    if n - p - 1 <= 0:
        return 0.0
        
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(classes)
    
    class_to_index = {c: i for i, c in enumerate(classes)}
    
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true, pred in zip(y_true, y_pred):
        matrix[class_to_index[true], class_to_index[pred]] += 1
        
    return matrix

def classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    
    total_samples = np.sum(cm)
    accuracy = np.trace(cm) / total_samples
    
    precisions = []
    recalls = []
    specificities = []
    f1_scores = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = total_samples - (tp + fp + fn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        f1_scores.append(f1)
        
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'specificity': np.mean(specificities),
        'f1_score': np.mean(f1_scores)
    }

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

def adjusted_r2_score(y_true, y_pred, num_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = num_features
    
    if n - p - 1 <= 0:
        return 0.0
    
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return adj_r2


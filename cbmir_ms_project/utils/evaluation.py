import numpy as np

def calculate_precision_at_k(retrieved_labels, query_label, k):
    """
    Calculates Precision@K.
    retrieved_labels: list of labels of the retrieved items.
    query_label: the true label of the query.
    """
    if k == 0:
        return 0.0
    relevant = sum(1 for label in retrieved_labels[:k] if label == query_label)
    return relevant / k

def calculate_recall_at_k(retrieved_labels, query_label, k, total_relevant_in_db):
    """
    Calculates Recall@K.
    """
    if total_relevant_in_db == 0:
        return 0.0
    relevant = sum(1 for label in retrieved_labels[:k] if label == query_label)
    return relevant / total_relevant_in_db

def calculate_average_precision(retrieved_labels, query_label, k):
    """
    Calculates Average Precision for a single query.
    """
    ap = 0.0
    relevant = 0
    for i in range(min(k, len(retrieved_labels))):
        if retrieved_labels[i] == query_label:
            relevant += 1
            precision_at_i = relevant / (i + 1)
            ap += precision_at_i
            
    if relevant == 0:
        return 0.0
    return ap / relevant

def evaluate_retrieval(query_label, retrieved_labels, total_relevant_in_db, k_values=[1, 3, 5, 10]):
    """
    Evaluates retrieval performance for a single query across multiple K values.
    """
    metrics = {}
    
    for k in k_values:
        if k > len(retrieved_labels):
            continue
            
        metrics[f'P@{k}'] = calculate_precision_at_k(retrieved_labels, query_label, k)
        metrics[f'R@{k}'] = calculate_recall_at_k(retrieved_labels, query_label, k, total_relevant_in_db)
        
    metrics['AP'] = calculate_average_precision(retrieved_labels, query_label, max(k_values))
    
    return metrics
    
if __name__ == "__main__":
    # Test
    q_label = 1 # Lesion
    ret_labels = [1, 0, 1, 1, 0]
    total_relevant = 10
    
    res = evaluate_retrieval(q_label, ret_labels, total_relevant, k_values=[3, 5])
    print("Metrics:", res)

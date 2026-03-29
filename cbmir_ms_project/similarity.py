import numpy as np
from scipy.spatial.distance import cdist, jensenshannon

def compute_distances(query_vector, dataset_vectors, metric='chebyshev'):
    """
    Computes distances between the query and the dataset using 1 of 9 available metrics.
    Returns an array of distances. Lower distance = Higher Similarity.
    
    Available metrics: 
    - euclidean
    - manhattan (cityblock)
    - chebyshev
    - cosine
    - mahalanobis
    - minkowski
    - braycurtis
    - canberra
    - jensenshannon
    """
    
    query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
    dataset_vectors = np.array(dataset_vectors, dtype=np.float32)
    
    valid_scipy_metrics = [
        'euclidean', 'cityblock', 'chebyshev', 'cosine', 
        'minkowski', 'braycurtis', 'canberra'
    ]
    
    # Handle naming aliases
    if metric == 'manhattan':
        metric_key = 'cityblock'
    else:
        metric_key = metric
        
    distances = np.zeros(len(dataset_vectors))
    
    if metric_key in valid_scipy_metrics:
        # Standard fast computation
        distances = cdist(query_vector, dataset_vectors, metric=metric_key)[0]
        
    elif metric_key == 'mahalanobis':
        # Requires the inverse covariance matrix of the dataset.
        # Fallback to euclidean if covariance is singular (e.g. dummy datasets).
        try:
            cov = np.cov(dataset_vectors, rowvar=False)
            inv_cov = np.linalg.pinv(cov)
            distances = cdist(query_vector, dataset_vectors, metric='mahalanobis', VI=inv_cov)[0]
        except:
            distances = cdist(query_vector, dataset_vectors, metric='euclidean')[0]
            
    elif metric_key == 'jensenshannon':
        # Jensen-Shannon requires probability distributions. 
        # We softmax the features to create pseudo-distributions.
        from scipy.special import softmax
        q_prob = softmax(query_vector[0])
        db_prob = softmax(dataset_vectors, axis=1)
        for i in range(len(dataset_vectors)):
            distances[i] = jensenshannon(q_prob, db_prob[i])
            
    else:
        raise ValueError(f"Unsupported metric: {metric}")
        
    return distances

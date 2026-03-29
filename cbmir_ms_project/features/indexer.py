import os
import numpy as np
import pickle
from scipy.spatial.distance import cdist

class CBMIRIndexer:
    def __init__(self, metric='chebyshev'):
        """
        Initializes the indexer for Content-Based Medical Image Retrieval.
        metric: 'chebyshev' (primary requirement) or 'euclidean'.
        """
        self.metric = metric
        self.metadata = []
        self.vectors = None
            
    def add_items(self, vectors, metadata_list):
        """
        Adds vectors and their corresponding metadata to the index.
        vectors: numpy array of shape [N, embedding_size]
        metadata_list: list of dicts, length N
        """
        if len(vectors) != len(metadata_list):
            raise ValueError(f"Vectors length ({len(vectors)}) must match metadata length ({len(metadata_list)})")
            
        vectors = np.array(vectors, dtype=np.float32)
        
        if self.vectors is None:
            self.vectors = vectors
            self.metadata = list(metadata_list)
        else:
            self.vectors = np.vstack((self.vectors, vectors))
            self.metadata.extend(metadata_list)
            
        print(f"Added {len(vectors)} items. Total in index: {len(self.metadata)}")

    def search(self, query_vector, k=10, metric_override=None):
        """
        Queries the index using the similarity metrics module.
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []
            
        # Use overriden metric (from UI dropdown) or default
        active_metric = metric_override if metric_override else self.metric
            
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from similarity import compute_distances
        
        distances = compute_distances(query_vector, self.vectors, metric=active_metric)
        
        # Sort by distance (lower = more similar)
        indices = np.argsort(distances)
        
        # Format results
        results = []
        # Exclude exact perfect matches (distance 0) if looking for other cases,
        # but for this assignment, we just return top k.
        for i in range(k):
            idx = indices[i]
            res = self.metadata[idx].copy()
            res['distance'] = distances[idx]
            results.append(res)
                
        return results

    def save(self, output_dir, prefix="cbmir"):
        """Saves the vectors and metadata to disk via numpy/pickle."""
        os.makedirs(output_dir, exist_ok=True)
        vec_path = os.path.join(output_dir, f"{prefix}_vectors.npy")
        meta_path = os.path.join(output_dir, f"{prefix}_meta.pkl")
        
        np.save(vec_path, self.vectors)
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        print(f"Saved vectors to {vec_path} and metadata to {meta_path}")

    def load(self, vec_path, meta_path):
        """Loads saved vectors and metadata."""
        if not os.path.exists(vec_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("Vectors or metadata file not found.")
            
        self.vectors = np.load(vec_path)
        
        with open(meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
            
        print(f"Loaded index with {len(self.metadata)} items using metric='{self.metric}'.")

if __name__ == "__main__":
    # Test
    indexer = CBMIRIndexer(metric='chebyshev')
    dummy_vecs = np.random.randn(100, 2048).astype(np.float32)
    meta = [{"id": i, "label": "test"} for i in range(100)]
    
    indexer.add_items(dummy_vecs, meta)
    
    q = np.random.randn(1, 2048).astype(np.float32)
    res = indexer.search(q, k=3)
    
    print(f"Top 3 results: {res}")

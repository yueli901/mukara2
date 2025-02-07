import tensorflow as tf
from config import MODEL

class LearnableClustering(tf.keras.layers.Layer):
    def __init__(self):
        super(LearnableClustering, self).__init__()
        
        self.num_clusters = MODEL['num_clusters']
        self.embedding_dim = MODEL['embedding_dim']
        
        # Learnable MLP for cluster assignment
        self.assignment_mlp = tf.keras.layers.Dense(self.num_clusters, activation='softmax')
    
    def call(self, pixel_embeddings):
        """
        Produce cluster embeddings based on O/D pixel embeddings of one sensor.

        Args:
            pixel_embeddings (tf.Tensor): (N, D) Pixel embeddings for one sensor.

        Returns:
            tuple: 
                - tf.Tensor: (K, D) cluster embeddings.
                - tf.Tensor: (N, K) cluster assignments.
        """
        # compute cluster assignment based on pixel embeddings
        cluster_assignments = self.assignment_mlp(pixel_embeddings)  # (N, K)
        
        # compute cluster embeddings
        weighted_sum = tf.matmul(cluster_assignments, pixel_embeddings, transpose_a=True)  # (K, D)
        sum_weights = tf.reduce_sum(cluster_assignments, axis=0)  # (K,)
        sum_weights = tf.expand_dims(sum_weights, axis=-1)  # (K, 1)
        sum_weights = tf.maximum(sum_weights, 1e-6)  # Prevent division by zero
        cluster_embeddings = weighted_sum / (sum_weights) # (K, D)
    
        assert cluster_embeddings.shape == (MODEL['num_clusters'], MODEL['embedding_dim']), "Shape mismatch!"

        return cluster_embeddings, cluster_assignments
import tensorflow as tf
from config import MODEL

class LearnableClustering(tf.keras.layers.Layer):
    def __init__(self):
        super(LearnableClustering, self).__init__()
        
        self.num_clusters = MODEL['num_clusters']
        self.embedding_dim = MODEL['embedding_dim']
        
        # MLP to project pixel embeddings into cluster space
        self.cluster_projection = tf.keras.layers.Dense(self.embedding_dim, activation='relu')
        
        # Learnable MLP for cluster assignment
        self.assignment_mlp = tf.keras.layers.Dense(self.num_clusters, activation='softmax')
    
    def compute_cluster_assignments(self, pixel_embeddings):
        """
        Compute soft cluster assignments for a **single sensor** using a learnable MLP.

        Args:
            pixel_embeddings (tf.Tensor): (N, D) pixel embeddings for one sensor.

        Returns:
            tf.Tensor: (N, K) cluster assignment probabilities.
        """
        projected_pixels = self.cluster_projection(pixel_embeddings)  # (N, D)
        cluster_assignments = self.assignment_mlp(projected_pixels)  # (N, K)
        
        return cluster_assignments  # (N, K)
    
    def compute_cluster_embeddings(self, pixel_embeddings, cluster_assignments):
        """
        Compute cluster embeddings for a **single sensor** as a weighted sum of pixel embeddings.

        Args:
            pixel_embeddings (tf.Tensor): (N, D) pixel embeddings for one sensor.
            cluster_assignments (tf.Tensor): (N, K) soft cluster assignment probabilities.

        Returns:
            tf.Tensor: (K, D) updated cluster embeddings.
        """
        weighted_sum = tf.matmul(cluster_assignments, pixel_embeddings, transpose_a=True)  # (K, D)
        sum_weights = tf.reduce_sum(cluster_assignments, axis=0)  # (K,)
        sum_weights = tf.expand_dims(sum_weights, axis=-1)  # (K, 1)
        sum_weights = tf.maximum(sum_weights, 1e-6)  # Prevent division by zero
        return weighted_sum / (sum_weights)
    
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
        cluster_assignments = self.compute_cluster_assignments(pixel_embeddings)  # (N, K)
        cluster_embeddings = self.compute_cluster_embeddings(pixel_embeddings, cluster_assignments)  # (K, D)
        
        assert cluster_embeddings.shape == (MODEL['num_clusters'], MODEL['embedding_dim']), "Shape mismatch!"

        return cluster_embeddings, cluster_assignments